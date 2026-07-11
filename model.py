import torch
import torch.nn as nn
from components import Resnet18
import random
import os
from res import ResNet, BasicBlock


class External(nn.Module):
    def __init__(self, res_dim):
        super(External, self).__init__()

        self.resnet = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=32)

        self.coefficient = nn.Sequential(
            nn.Linear(res_dim, 1), nn.LeakyReLU(negative_slope=0.2),
        )
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

    def forward(self, node, adjacency):    # Conv2d[n,c,h,w]
        n = adjacency.size(0)
        node_feature = self.resnet(node.permute(2, 0, 1).unsqueeze(0))
        output = node_feature.repeat(n, 1)
        return output


class Pressure(nn.Module):
    def __init__(self, node_input_dim, node_embedding_dim):
        super(Pressure, self).__init__()

        self.node_embedding = nn.Linear(node_input_dim, node_embedding_dim)
        self.coefficient = nn.Sequential(
            nn.Linear(node_embedding_dim * 2, 1),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

    def forward(self, node, adjacency, grid_n):  # [n,n]
        n = node.size()[0]

        node_other = self.node_embedding(
            node.repeat(n, 1).view(n, n, -1)
        )
        node_self = self.node_embedding(
            node.repeat(n, 1).view(n, n, -1).transpose(0, 1)
        )

        alpha = self.coefficient(
            torch.cat([node_other, node_self], dim=-1)
        )
        alpha = self.softmax(alpha * adjacency.unsqueeze(-1))

        grid_n_other = grid_n.repeat(n, 1).view(n, n, -1)
        grid_n_self = grid_n_other.transpose(0, 1)
        strength = torch.cosine_similarity(grid_n_other, grid_n_self, dim=-1) * adjacency

        output = self.relu(alpha * strength.unsqueeze(-1) * node_other)
        output = torch.sum(output, dim=1)
        return output


class Viscosity(nn.Module):
    def __init__(self, node_input_dim, node_embedding_dim):
        super(Viscosity, self).__init__()

        self.node_embedding = nn.Linear(node_input_dim, node_embedding_dim)
        self.coefficient = nn.Sequential(
            nn.Linear(node_embedding_dim * 2, 1),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.density = nn.Sequential(
            nn.Linear(2, node_embedding_dim),
            nn.ReLU()
        )
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

    def forward(self, node, adjacency, grid_v):  # [n,n]
        n = node.size()[0]

        node_other = self.node_embedding(
            node.repeat(n, 1).view(n, n, -1)
        )
        node_self = self.node_embedding(
            node.repeat(n, 1).view(n, n, -1).transpose(0, 1)
        )
        grid_v_other = grid_v.repeat(n, 1).view(n, n, -1)

        alpha = self.coefficient(
            torch.cat([node_other, node_self], dim=-1)
        )
        alpha = self.softmax(alpha * adjacency.unsqueeze(-1))
        node_select = self.density(grid_v_other) * node_other
        output = self.relu(alpha * node_select)
        output = torch.sum(output, dim=1)
        return output


class DecoupleBlock(nn.Module):
    def __init__(self, node_dim, ):  # layer_dim
        super(DecoupleBlock, self).__init__()

        self.embedding = nn.Linear(node_dim, node_dim)
        self.coefficient = nn.Sequential(
            nn.Linear(node_dim * 2, 1),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

    def forward(self, node, adjacency):
        n = node.size()[0]

        node_other = self.embedding(
            node.repeat(n, 1).view(n, n, -1)
        )
        node_self = self.embedding(
            node.repeat(n, 1).view(n, n, -1).transpose(0, 1)
        )

        alpha = self.coefficient(
            torch.cat([node_other, node_self], dim=-1)
        )
        alpha = self.softmax(alpha * adjacency.unsqueeze(-1))

        output = self.relu(alpha * node_other)
        output = torch.sum(output, dim=1)
        return output


class Decouple(nn.Module):
    def __init__(self, layer_num, decouple_dim):
        super(Decouple, self).__init__()

        self.layer_stack = nn.ModuleList()
        for i in range(layer_num):
            self.layer_stack.append(DecoupleBlock(node_dim=decouple_dim))

    def forward(self, node, adjacency):
        force = node
        for i, layer in enumerate(self.layer_stack):
            node = node - layer(node, adjacency)

        res = force - node
        return node, res


class Prediction(nn.Module):
    def __init__(
            self, obs_len, pred_len, encoder_lstm_hidden_dim, decoder_lstm_hidden_dim,
            n_embedding_dim, v_embedding_dim,
            pressure_dim, viscosity_dim, res_dim,
            layer_num, decouple_dim
    ):
        super(Prediction, self).__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.encoder_lstm_hidden_dim = encoder_lstm_hidden_dim
        self.v_embedding_dim = v_embedding_dim
        self.pressure_dim = pressure_dim

        self.input_embedding_v = nn.Linear(2, v_embedding_dim)

        self.lstm_e = nn.LSTMCell(v_embedding_dim, encoder_lstm_hidden_dim)

        self.pressure = Pressure(node_input_dim=encoder_lstm_hidden_dim, node_embedding_dim=pressure_dim)
        self.viscosity = Viscosity(node_input_dim=encoder_lstm_hidden_dim, node_embedding_dim=viscosity_dim)
        self.external = External(res_dim=res_dim)

        # self.fusion = nn.Linear(pressure_dim + viscosity_dim, decouple_dim)     # + external_output_dim

        self.decouple = Decouple(layer_num=layer_num, decouple_dim=decouple_dim)

        self.lstm_d = nn.LSTMCell(v_embedding_dim, decouple_dim)

        self.h2v = nn.Linear(decouple_dim, 2)
        self.h2n = nn.Linear(decouple_dim, 1)

        self.relu = nn.ReLU()
        self.init()

    def init(self):
        nn.init.kaiming_normal_(self.lstm_d.weight_hh)
        nn.init.kaiming_normal_(self.lstm_d.weight_ih)
        nn.init.zeros_(self.lstm_d.bias_hh)
        nn.init.zeros_(self.lstm_d.bias_ih)
        nn.init.kaiming_normal_(self.lstm_e.weight_hh)
        nn.init.kaiming_normal_(self.lstm_e.weight_ih)
        nn.init.zeros_(self.lstm_e.bias_hh)
        nn.init.zeros_(self.lstm_e.bias_ih)

    def t_encoding(self, inputs):
        ped_num = inputs.size(1)
        hidden_state = torch.randn(ped_num, self.encoder_lstm_hidden_dim).cuda()
        cell_state = torch.randn(ped_num, self.encoder_lstm_hidden_dim).cuda()

        inputs_embedding = self.relu(self.input_embedding_v(inputs.contiguous().view(-1, 2)))
        inputs_embedding = inputs_embedding.view(-1, ped_num, self.v_embedding_dim)

        for i, input_data in enumerate(
                inputs_embedding[:self.obs_len].chunk(
                    inputs_embedding[:self.obs_len].size(0), dim=0
                )
        ):
            hidden_state, cell_state = self.lstm_e(
                input_data.squeeze(0), (hidden_state, cell_state)
            )

        return hidden_state  # [ped_num, encoder_lstm_hidden_dim]

    def decoding_one(self, grid_v, grid_n_curr, teacher_ratio, adjacency, grid_v_encoded, scene):
        pressure_force = self.pressure(grid_v_encoded, adjacency, grid_n_curr.unsqueeze(-1))
        viscosity_force = self.viscosity(grid_v_encoded, adjacency, grid_v[self.obs_len - 1])
        external_force = self.external(scene, adjacency)

        force = pressure_force + viscosity_force + external_force  #

        pred_n = []
        pred_v = []
        output_v = grid_v[self.obs_len - 1]

        hidden_state = force
        cell_state = torch.zeros_like(hidden_state).cuda()

        grid_v_embedded = self.relu(self.input_embedding_v(grid_v.contiguous().view(-1, 2)))
        grid_v_embedded = grid_v_embedded.view(-1, grid_v.size(1), self.v_embedding_dim)

        if self.training:
            for t in range(self.pred_len):
                if random.random() < teacher_ratio:
                    input_data = grid_v_embedded[-self.pred_len + t]
                else:
                    input_data = self.relu(self.input_embedding_v(output_v))

                if t == 0:
                    hidden_state, res = self.decouple(hidden_state, adjacency)

                hidden_state, cell_state = self.lstm_d(
                    input_data.squeeze(0), (hidden_state, cell_state)
                )

                output_v = self.h2v(hidden_state)
                pred_v.append(output_v)

        else:
            for t in range(self.pred_len):
                input_data = self.relu(self.input_embedding_v(output_v))

                if t == 0:
                    hidden_state, res = self.decouple(hidden_state, adjacency)

                hidden_state, cell_state = self.lstm_d(
                    input_data.squeeze(0), (hidden_state, cell_state)
                )

                output_v = self.h2v(hidden_state)
                pred_v.append(output_v)

        return torch.stack(pred_v)

    def forward(self, grid_n, grid_v, adjacency, scene, teacher_ratio=0.5):

        grid_n = grid_n.view(grid_n.size(0), -1)
        grid_v = grid_v.view(grid_n.size(0), -1, 2)

        grid_v_encoded = self.t_encoding(grid_v)

        pred_v = self.decoding_one(
            grid_v, grid_n[self.obs_len - 1], teacher_ratio, adjacency, grid_v_encoded, scene
        )

        return pred_v




