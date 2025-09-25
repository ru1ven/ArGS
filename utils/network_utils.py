import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.pointnet_utils import PointNetSetAbstraction


class PointNet2(nn.Module):
    def __init__(self, in_channel=3, out_pts=1, pretrained_path='../lib/pretrained/pointnet2_ssg.pt', pre_trained=True):
        super(PointNet2, self).__init__()
        self.out_pts = out_pts
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128],
                                          group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256],
                                          group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        # load pre-trained weight
        if pre_trained:
            latest_checkpoint = torch.load(pretrained_path)
            filtered_state_dict = {k.replace('module.point_encoder.', ''): v for k, v in
                                   latest_checkpoint['state_dict'].items() if k.startswith('module.point_encoder')}
            self.load_state_dict(filtered_state_dict)

        # embedding the input channel
        if out_pts != 1:
            self.sa4 = PointNetSetAbstraction(npoint=2, radius=0.6, nsample=96, in_channel=256 + 3, mlp=[256, 512, 1024],
                                          group_all=False)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        fea = xyz[:, 3:, :]
        xyz = xyz[:, :3, :]

        l1_xyz, l1_points,_ = self.sa1(xyz, fea)
        l2_xyz, l2_points,_ = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points,_ = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, self.out_pts, 1024)
        # x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        # x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x

class Pointnet2_Ssg(nn.Module):
    def __init__(self, channel_pc, normal_channel=False, pretrained_path='../lib/pretrained/pointnet2_ssg.pt', pre_trained=True):
        super(Pointnet2_Ssg, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        # self.fc3 = nn.Linear(256, num_class)

        # load pre-trained weight
        if pre_trained:
            latest_checkpoint = torch.load(pretrained_path)
            filtered_state_dict = {k.replace('module.point_encoder.',''): v for k, v in latest_checkpoint['state_dict'].items() if k.startswith('module.point_encoder')}
            self.load_state_dict(filtered_state_dict)

        # embedding the input channel
        self.sa1.mlp_convs[0] = nn.Conv2d(channel_pc, 64, 1)


    def forward(self, xyz):
        B, _, _ = xyz.shape

        norm = xyz[:, 3:, :]
        xyz = xyz[:, :3, :]

        l1_xyz, l1_points, fps = self.sa1(xyz, norm)
        # l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # x = l3_points.view(B, 1024)
        #
        # # x = F.relu(self.fc1(x))
        # # x = F.relu(self.fc2(x))
        # x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        # x = self.drop2(F.relu(self.bn2(self.fc2(x))))

        return l1_points, fps


    def forward_unit(self, xyz):

        norm = xyz[:, 3:, :]
        xyz = xyz[:, :3, :]

        l1_xyz, l1_points, fps_hand, fps_obj = self.sa1.forward_unit(xyz, norm)

        B, _, N = l1_points.shape

        return l1_points[:, :, :N//2], l1_points[:, :, N//2:], fps_hand, fps_obj-N//2


class PointMLP(nn.Module):
    def __init__(self, joint_num=50, points=1024, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=True, use_xyz=True, normalize="anchor",
                 dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[16, 16, 16, 16], reducers=[4, 4, 4, 4],
                 de_dims=[512, 256, 128, 128], de_blocks=[4, 4, 4, 4],
                 gmp_dim=64, **kwargs):
        super(PointMLP, self).__init__()
        self.stages = len(pre_blocks)
        self.joint_num = joint_num
        self.points = points
        self.embedding = ConvBNReLU1D(3, embed_dim, bias=bias, activation=activation)
        assert len(pre_blocks) == len(k_neighbors) == len(reducers) == len(pos_blocks) == len(dim_expansion), \
            "Please check stage number consistent for pre_blocks, pos_blocks k_neighbors, reducers."
        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()
        last_channel = embed_dim
        anchor_points = self.points
        en_dims = [last_channel]
        ### Building Encoder #####
        for i in range(len(pre_blocks)):
            out_channel = last_channel * dim_expansion[i]
            pre_block_num = pre_blocks[i]
            pos_block_num = pos_blocks[i]
            kneighbor = k_neighbors[i]
            reduce = reducers[i]
            anchor_points = anchor_points // reduce
            # append local_grouper_list
            local_grouper = LocalGrouper(last_channel, anchor_points, kneighbor, use_xyz, normalize)  # [b,g,k,d]
            self.local_grouper_list.append(local_grouper)
            # append pre_block_list
            pre_block_module = PreExtraction(last_channel, out_channel, pre_block_num, groups=groups,
                                             res_expansion=res_expansion,
                                             bias=bias, activation=activation, use_xyz=use_xyz)
            self.pre_blocks_list.append(pre_block_module)
            # append pos_block_list
            pos_block_module = PosExtraction(out_channel, pos_block_num, groups=groups,
                                             res_expansion=res_expansion, bias=bias, activation=activation)
            self.pos_blocks_list.append(pos_block_module)

            last_channel = out_channel
            en_dims.append(last_channel)


        ### Building Decoder #####
        self.decode_list = nn.ModuleList()
        en_dims.reverse()
        de_dims.insert(0,en_dims[0])
        assert len(en_dims) ==len(de_dims) == len(de_blocks)+1
        for i in range(len(en_dims)-1):
            self.decode_list.append(
                PointNetFeaturePropagation(de_dims[i]+en_dims[i+1], de_dims[i+1],
                                           blocks=de_blocks[i], groups=groups, res_expansion=res_expansion,
                                           bias=bias, activation=activation)
            )

        self.act = get_activation(activation)

        # global max pooling mapping
        self.gmp_map_list = nn.ModuleList()
        for en_dim in en_dims:
            self.gmp_map_list.append(ConvBNReLU1D(en_dim, gmp_dim, bias=bias, activation=activation))
        self.gmp_map_end = ConvBNReLU1D(gmp_dim*len(en_dims), gmp_dim, bias=bias, activation=activation)

        # classifier
        self.conv = nn.Sequential(
            nn.Conv1d(gmp_dim+de_dims[-1], 128, 1, bias=bias),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        out_dims = [joint_num*3, joint_num, joint_num]
        self.finals = nn.ModuleList()
        for out_dim in out_dims:
            self.finals.append(nn.Conv1d(in_channels=128, out_channels=out_dim, kernel_size=1, stride=1))

        self.en_dims = en_dims

    def forward(self, x):
        xyz = x.permute(0, 2, 1)
        pcl = xyz.clone()
        x = self.embedding(x)  # B,D,N

        xyz_list = [xyz]  # [B, N, 3]
        x_list = [x]  # [B, D, N]

        # here is the encoder
        for i in range(self.stages):
            # Give xyz[b, p, 3] and fea[b, p, d], return new_xyz[b, g, 3] and new_fea[b, g, k, d]
            xyz, x = self.local_grouper_list[i](xyz, x.permute(0, 2, 1))  # [b,g,3]  [b,g,k,d]
            x = self.pre_blocks_list[i](x)  # [b,d,g]
            x = self.pos_blocks_list[i](x)  # [b,d,g]
            xyz_list.append(xyz)
            x_list.append(x)

        # here is the decoder
        xyz_list.reverse()
        x_list.reverse()
        x = x_list[0]
        for i in range(len(self.decode_list)):
            x = self.decode_list[i](xyz_list[i+1], xyz_list[i], x_list[i+1],x)

        # here is the global context
        gmp_list = []
        for i in range(len(x_list)):
            gmp_list.append(F.adaptive_max_pool1d(self.gmp_map_list[i](x_list[i]), 1))
        global_context = self.gmp_map_end(torch.cat(gmp_list, dim=1)) # [b, gmp_dim, 1]

        #here is the cls_token
        x = torch.cat([x, global_context.repeat([1, 1, x.shape[-1]])], dim=1)
        device = x.device
        point_feature = self.conv(x)  # (batch_size, 256, num_points) -> (batch_size, 13, num_points)
        point_result = torch.Tensor().to(device)
        for layer in self.finals:
            temp = layer(point_feature)
            point_result = torch.cat((point_result, temp), dim=1)
        return [[pcl, point_result.permute(0,2,1)]]