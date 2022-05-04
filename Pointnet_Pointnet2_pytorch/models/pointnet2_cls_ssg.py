import torch.nn as nn
import torch.nn.functional as F
from pointnet_util import PointNetSetAbstraction


class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        # 512 = points sampled in farthest point sampling
        # 0.2 = search radius in local region
        # 32 = how many points in each local region
        # [64,64,128] = output size for MLP on each point 
        # 3 = 3-dim coordinates
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        # fc1 input:1024
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        # fc2 input:512
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        # fc3 input:256
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None

        # l1_points作为sa1的特征输出
        l1_xyz, l1_points = self.sa1(xyz, norm)
        # l2_points作为sa2的特征输出
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # l3_points作为sa3的特征输出
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1) # 计算对数概率


        return x, l3_points



class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        # NLLLoss的输入是一个对数概率向量和一个目标标签. 它不会计算对数概率. 
        # 适合网络的最后一层是log_softmax. 
        # 损失函数 nn.CrossEntropyLoss()与NLLLoss()相同, 唯一的不同是它去做softmax.
        total_loss = F.nll_loss(pred, target)

        return total_loss
