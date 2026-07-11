import os
import logging
import torch
import cv2
import shutil


def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
        )
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    if is_best:
        torch.save(state, filename)
        logging.info("-------------- lower ade ----------------")
        shutil.copyfile(filename, "model_best.pth.tar")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def cal_loss(pred, grid_v):        # grid_v [t,x,y,2]
    t, x, y, _ = grid_v.size()
    gt = grid_v.view(t, -1, 2)

    e = torch.arange(t).float().cuda()
    e = torch.exp(e / 6)

    loss = torch.sum(torch.sum(((pred - gt) ** 2), dim=-1), dim=-1)
    loss = loss * e
    loss = torch.sum(loss)
    return loss


def cal_val(pred, grid_v):
    t, x, y, _ = grid_v.size()
    gt = grid_v.view(t, -1, 2)
    loss = torch.sum(torch.sum(((pred - gt) ** 2), dim=-1), dim=-1)
    ade = torch.mean(loss)
    fde = loss[-1]
    return ade, fde


def cal_test(pred, grid_v):
    t, x, y, _ = grid_v.size()
    gt = grid_v.view(t, -1, 2)
    loss = torch.sum(torch.sum(((pred - gt) ** 2), dim=-1), dim=-1)
    ade = torch.mean(loss)
    fde = loss[-1]
    return ade, fde


def data_process(traj, start_end, scene_path):
    scene = cv2.imread(scene_path)
    scene_y_max, scene_x_max, _ = scene.shape

    seq_len, num_ped, _ = traj.size()

    velocity = traj[1:, :, :] - traj[:-1, :, :]

    ped_x_max = torch.max(torch.abs(traj[:-1, :, 0] - traj[1:, :, 0]))
    ped_y_max = torch.max(torch.abs(traj[:-1, :, 1] - traj[1:, :, 1]))

    num_x = torch.floor(scene_x_max / ped_x_max).int().item()
    num_y = torch.floor(scene_y_max / ped_y_max).int().item()
    len_x = scene_x_max / num_x
    len_y = scene_y_max / num_y

    grid = torch.zeros([seq_len, num_ped, 2])
    grid[:, :, 0] = torch.floor(traj[:, :, 0] / len_x)
    grid[:, :, 1] = torch.floor(traj[:, :, 1] / len_y)

    grid_n = torch.zeros([seq_len, num_x, num_y]).cuda()
    grid_v = torch.zeros([seq_len, num_x, num_y, 2]).cuda()
    grid_a = torch.zeros([seq_len, num_x, num_y, 2]).cuda()

    for t in range(seq_len):
        for id in range(num_ped):
            idx, idy = grid[t, id, 0].int().item(), grid[t, id, 1].int().item()
            grid_n[t, idx, idy] += 1
            if t > 0:
                grid_v[t, idx, idy, :] += velocity[t-1, id, :]
            if t > 1:
                grid_a[t, idx, idy, :] += (velocity[t-1, id, :] - velocity[t-2, id, :])

    return grid_n, grid_v, grid_a


def get_scene(path):
    img = cv2.imread(path)
    scene = torch.from_numpy(img).float().cuda()
    return scene

