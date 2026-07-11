import argparse
import logging
import numpy as np
import os
import random
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time

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
    "--test_path", type=str,
    default="./datasets/ds/1/test"
)
parser.add_argument(
    "--scene_path", type=str,
    default="./datasets/ds/1/1.jpg"
)

parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--seed", type=int, default=72, help="Random seed.")
parser.add_argument("--num_samples", default=1, type=int)
parser.add_argument("--gpu_num", default="0", type=str)
parser.add_argument(
    "--resume", type=str,
    default="",
)


def get_generator(checkpoint):
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
    model.load_state_dict(checkpoint["state_dict"])
    model.cuda()
    model.eval()
    return model


def evaluate(args, test_loader, generator, scene):
    ADE, FDE = [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch = [tensor.cuda() for tensor in batch]
            (grid_n, grid_v, adjacency) = batch

            for _ in range(args.num_samples):
                start = time.time()
                pred_v = generator(grid_n.squeeze(0), grid_v.squeeze(0), adjacency, scene)
                torch.cuda.synchronize()
                end = time.time()
                print(end-start)

                ade_, fde_ = utils.cal_test(pred_v, grid_v.squeeze(0)[args.obs_len:])
                ADE.append(ade_)
                FDE.append(fde_)

    ade = sum(ADE) / len(ADE)
    fde = sum(FDE) / len(FDE)
    return ade, fde


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

    checkpoint = torch.load(args.resume)
    generator = get_generator(checkpoint)

    test_dset, test_loader = data_loader(args, args.test_path)
    scene = utils.get_scene(args.scene_path)

    ade, fde = evaluate(args, test_loader, generator, scene)

    print(
        "Dataset: Pred Len: {}, ADE: {:.12f}, FDE: {:.12f}".format(
            args.pred_len, ade, fde
        )
    )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
