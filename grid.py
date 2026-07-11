import os
import numpy as np
import cv2
import math
import torch
from torch.utils.data import Dataset


def grid_collate(data):
    (grid_n_list, grid_v_list, adjacency) = zip(*data)
    grid_n = torch.cat(grid_n_list, dim=0)
    grid_v = torch.cat(grid_v_list, dim=0)
    out = [grid_n, grid_v, adjacency[0]]
    return tuple(out)


class GridDataset(Dataset):
    def __init__(self, path, scene_path, obs_len=8, pred_len=12, delim=","):
        super(GridDataset, self).__init__()
        self.path = path
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim

        file_path = os.listdir(self.path)
        all_file = [os.path.join(self.path, _path) for _path in file_path]

        read_data = []
        num_len = [0]
        for file in all_file:
            with open(file, "r") as f:
                for line in f:
                    line = line.strip().split(self.delim)
                    line = [float(i) for i in line]
                    read_data.append(line)
            num_len.append(len(read_data))

        read_data = np.asarray(read_data)
        data = read_data[:, :4]
        data[:, 0] = read_data[:, 0] - 1
        data[:, 2] = read_data[:, 2] + read_data[:, 4] / 2
        data[:, 3] = read_data[:, 3] + read_data[:, 5] / 2

        scene = cv2.imread(scene_path)
        scene_y_max, scene_x_max, _ = scene.shape

        traj = self.get_traj(all_file)      #
        ped_x_max = np.max(np.abs(traj[:, 0, :-1] - traj[:, 0, 1:]))
        ped_y_max = np.max(np.abs(traj[:, 1, :-1] - traj[:, 1, 1:]))
        # ped_x_max = 60.4
        # ped_y_max = 39.4

        num_x = math.floor(scene_x_max / ped_x_max)
        num_y = math.floor(scene_y_max / ped_y_max)

        len_x = scene_x_max / num_x
        len_y = scene_y_max / num_y

        grid_n_20_all = []
        grid_v_20_all = []
        for f in range(len(all_file)):
            f_data = data[num_len[f]:num_len[f+1], :]

            num_t = np.max(f_data).astype(int)

            grid_n_, grid_v_ = self.file(num_t, num_x, num_y, f_data, len_x, len_y)
            grid_n_20_all.append(grid_n_)
            grid_v_20_all.append(grid_v_)

        self.grid_n = torch.from_numpy(np.concatenate(grid_n_20_all, axis=0)).type(torch.float)
        self.grid_v = torch.from_numpy(np.concatenate(grid_v_20_all, axis=0)).type(torch.float)
        self.seq_len = self.grid_n.shape[0]

        adj = self.get_adjacency(self.grid_n.shape[2], self.grid_n.shape[3])
        self.adjacency = torch.from_numpy(adj).type(torch.float)

    def file(self, num_t, num_x, num_y, data, len_x, len_y):
        grid_n = np.zeros([num_t, num_x, num_y])
        grid_v = np.zeros([num_t, num_x, num_y, 2])

        for n in range(data.shape[0]):
            t, id, x, y = int(data[n, 0]), data[n, 1], data[n, 2], data[n, 3]

            grid_n[t, math.floor(x / len_x), math.floor(y / len_y)] += 1
            if t > 1:
                n_m = np.argwhere((data[:, 0] == (t-1)) & (data[:, 1] == id))
                if n_m.size == 0:
                    continue
                n_m = n_m[0, 0]
                x_m, y_m = data[n_m, 2:4]
                if math.floor(x_m / len_x) != math.floor(x / len_x) or math.floor(y_m / len_y) != math.floor(y / len_y):
                    grid_v[t, math.floor(x / len_x), math.floor(y / len_y), 0] += 1    # i

                n_p = np.argwhere((data[:, 0] == (t+1)) & (data[:, 1] == id))
                if n_p.size == 0:
                    continue
                n_p = n_p[0, 0]
                x_p, y_p = data[n_p, 2:4]
                if math.floor(x_p / len_x) != math.floor(x / len_x) or math.floor(y_p / len_y) != math.floor(y / len_y):
                    grid_v[t, math.floor(x / len_x), math.floor(y / len_y), 1] += 1  # o

        grid_n_all = torch.from_numpy(grid_n).type(torch.float)
        grid_v_all = torch.from_numpy(grid_v).type(torch.float)

        grid_n_20 = torch.zeros(num_t-19, 20, num_x, num_y)
        grid_v_20 = torch.zeros(num_t-19, 20, num_x, num_y, 2)
        for t in range(num_t-19):
            grid_n_20[t, :, :, :] = grid_n_all[t:t+20, :, :]
            grid_v_20[t, :, :, :, :] = grid_v_all[t:t+20, :, :, :]
        return grid_n_20, grid_v_20

    def get_adjacency(self, num_x, num_y):
        adjacency = np.zeros([num_x * num_y, num_x * num_y])
        for x in range(num_x):
            for y in range(num_y):
                if x + 1 < num_x:
                    adjacency[x * num_y + y, (x + 1) * num_y + y] = 1
                if x - 1 > 0:
                    adjacency[x * num_y + y, (x - 1) * num_y + y] = 1
                if y + 1 < num_y:
                    adjacency[x * num_y + y, x * num_y + y + 1] = 1
                if y - 1 > 0:
                    adjacency[x * num_y + y, x * num_y + y - 1] = 1
        return adjacency

    def get_traj(self, all_file):
        seq_list = []
        ped_num = []

        for file in all_file:
            read_data = []
            with open(file, "r") as f:
                for line in f:
                    line = line.strip().split(self.delim)
                    line = [float(i) for i in line]
                    read_data.append(line)
            read_data = np.asarray(read_data)
            data = read_data[:, :4]
            data[:, 0] = read_data[:, 0] - 1
            data[:, 2] = read_data[:, 2] + read_data[:, 4] / 2
            data[:, 3] = read_data[:, 3] + read_data[:, 5] / 2

            frames = np.unique(data[:, 0]).tolist()
            traj_num = int(math.ceil(len(frames) - self.seq_len + 1))
            frame_data = []
            for f in frames:
                frame_data.append(data[f == data[:, 0], :])

            for idx in range(traj_num):
                curr_traj = np.concatenate(frame_data[idx:idx+self.seq_len], axis=0)
                ped = np.unique(curr_traj[:, 1])

                curr_traj_process = np.zeros((len(ped), 2, self.seq_len))
                ped_considered = 0
                for _, id in enumerate(ped):
                    curr_traj_ped = curr_traj[curr_traj[:, 1] == id, :]
                    curr_traj_ped = np.around(curr_traj_ped, decimals=4)

                    ped_front = frames.index(curr_traj_ped[0, 0]) - idx
                    ped_end = frames.index(curr_traj_ped[-1, 0]) - idx + 1
                    if ped_end - ped_front != self.seq_len or np.shape(curr_traj_ped)[0] != self.seq_len:
                        continue

                    curr_traj_ped = np.transpose(curr_traj_ped[:, 2:])
                    _i = ped_considered
                    curr_traj_process[_i, :, ped_front:ped_end] = curr_traj_ped
                    ped_considered = ped_considered + 1

                seq_list.append(curr_traj_process[:ped_considered])
                ped_num.append(ped_considered)

        self.seq_num = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        # self.traj = torch.from_numpy(seq_list).type(torch.float)
        return seq_list

    def __len__(self):
        return self.seq_len

    def __getitem__(self, item):
        out = [self.grid_n[item].unsqueeze(0), self.grid_v[item].unsqueeze(0), self.adjacency]
        return out


