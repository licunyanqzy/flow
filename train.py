import argparse
import logging
import numpy as np
import os
import random
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DataParallel
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as ddp
from torch.utils.tensorboard import SummaryWriter

from model import Prediction
from loader import data_loader
import utils


parser = argparse.ArgumentParser()

parser.add_argument("--delim", default=",")
parser.add_argument("--loader_num_workers", default=1, type=int)
parser.add_argument("--obs_len", default=8, type=int)
parser.add_argument("--pred_len", default=12, type=int)

parser.add_argument("--encoder_lstm_hidden_dim", type=int, default=32)
parser.add_argument("--decoder_lstm_hidden_dim", type=int, default=32)
parser.add_argument("--n_embedding_dim", type=int, default=16)
parser.add_argument("--v_embedding_dim", type=int, default=16)
parser.add_argument("--a_embedding_dim", type=int, default=16)
parser.add_argument("--pressure_dim", type=int, default=32)
parser.add_argument("--viscosity_dim", type=int, default=32)
parser.add_argument("--res_dim", type=int, default=32)
parser.add_argument("--decouple_dim", type=int, default=32)
parser.add_argument("--decouple_layer_num", type=int, default=1)

parser.add_argument(
    "--train_path", type=str,
    default="./datasets/ds/1/train"
)
parser.add_argument(
    "--val_path", type=str,
    default="./datasets/ds/1/test"
)
parser.add_argument(
    "--scene_path", type=str,
    default="./datasets/ds/1/1.jpg"    # 1 2 3
)
parser.add_argument("--checkpoint_dir", default="./checkpoint_1", type=str)
parser.add_argument("--log_dir", default="./")

parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--seed", type=int, default=72, help="Random seed.")
parser.add_argument("--gpu_num", default="0", type=str)
parser.add_argument("--lr", default=1e-3, type=int)
parser.add_argument("--start_epoch", default=0, type=int)
parser.add_argument("--num_epoch", default=400, type=int)
parser.add_argument("--best_k", default=10, type=int)
parser.add_argument("--print_every", default=1000, type=int)
parser.add_argument("--resume", default="", type=str)   # ./checkpoint


bestADE = 1000000000.0


def train(args, model, train_loader, optimizer, epoch, writer, scene):
    losses = utils.AverageMeter("Loss", ":.6f")
    progress = utils.ProgressMeter(
        len(train_loader), [losses], prefix="Epoch: [{}]".format(epoch)
    )

    model.train()
    for batch_idx, batch in enumerate(train_loader):
        batch = [tensor.cuda() for tensor in batch]
        (grid_n, grid_v, adjacency) = batch

        optimizer.zero_grad()
        loss = torch.zeros(1).cuda()
        teacher_ratio = np.exp(-(epoch) / 20)

        pred_v = model(grid_n.squeeze(0), grid_v.squeeze(0), adjacency, scene, teacher_ratio)

        loss += utils.cal_loss(pred_v, grid_v.squeeze(0)[args.obs_len:])

        losses.update(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, norm_type=2)
        optimizer.step()

        if batch_idx % args.print_every == 0:
            progress.display(batch_idx)

    writer.add_scalar("train_loss", losses.avg, epoch)


def validate(args, model, val_loader, epoch, writer, scene):
    ADE = utils.AverageMeter("ADE", ":.6f")
    FDE = utils.AverageMeter("FDE", ":.6f")
    progress = utils.ProgressMeter(len(val_loader), [ADE, FDE], prefix="Test: ")

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            batch = [tensor.cuda() for tensor in batch]
            (grid_n, grid_v, adjacency) = batch

            pred_v = model(grid_n.squeeze(0), grid_v.squeeze(0), adjacency, scene)

            ade, fde = utils.cal_val(pred_v, grid_v.squeeze(0)[args.obs_len:])
            ADE.update(ade, grid_v.size(0))
            FDE.update(fde, grid_v.size(0))

            if batch_idx % args.print_every == 0:
                progress.display(batch_idx)

        logging.info(
            "* ADE {ade.avg:.3f} FDE {fde.avg:.3f}".format(ade=ADE, fde=FDE)
        )
        writer.add_scalar("val_ade", ADE.avg, epoch)

    return ADE.avg


def main(args):
    utils.set_logger(os.path.join(args.log_dir, "train.log"))
    if os.path.exists(args.checkpoint_dir) is False:
        os.mkdir(args.checkpoint_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

    logging.info("Initializing train dataset")
    train_dset, train_loader = data_loader(args, args.train_path)
    logging.info("Initializing val dataset")
    val_dset, val_loader = data_loader(args, args.val_path)
    logging.info("Initializing scene")
    scene = utils.get_scene(args.scene_path)

    model = Prediction(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        encoder_lstm_hidden_dim=args.encoder_lstm_hidden_dim,
        decoder_lstm_hidden_dim=args.decoder_lstm_hidden_dim,
        n_embedding_dim=args.n_embedding_dim,
        v_embedding_dim=args.v_embedding_dim,
        pressure_dim=args.pressure_dim,
        viscosity_dim=args.viscosity_dim,
        res_dim=args.res_dim,
        layer_num=args.decouple_layer_num,
        decouple_dim=args.decouple_dim,
    )
    model.cuda()
    parameters = ([p for p in model.parameters()])
    optimizer = optim.Adam(parameters, lr=args.lr)

    writer = SummaryWriter()

    if args.resume:     # start from checkpoint
        if os.path.isfile(args.resume):
            logging.info("Restoring from checkpoint {}".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            logging.info(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            logging.info("=> no checkpoint found as '{}'".format(args.resume))

    global bestADE

    for epoch in range(args.start_epoch, args.num_epoch):
        train(args, model, train_loader, optimizer, epoch, writer, scene)
        ADE = validate(args, model, val_loader, epoch, writer, scene)

        # if local_rank == 0:
        is_best = ADE < bestADE
        bestADE = min(ADE, bestADE)
        utils.save_checkpoint(  # if ADE > bestADE, save checkpoint
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_ADE": bestADE,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            f"./checkpoint_1/checkpoint{epoch}.pth.tar",
        )

    writer.close()


if __name__ == "__main__":
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    args = parser.parse_args()
    main(args)




