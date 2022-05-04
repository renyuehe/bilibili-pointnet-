import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

# STN3d: T-Net 3*3 transform
# 类似一个mini-PointNet
class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        # torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9) # 9=3*3
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # Symmetric function: max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        # x参数展平（拉直）
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # 展平的对角矩阵：np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden # affine transformation
        # 用view，转换成batchsize*3*3的数组
        x = x.view(-1, 3, 3)
        return x


# STNkd: T-Net 64*64 transform，k默认是64
class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # Symmetric function: max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        # 参数拉直（展平）
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # 展平的对角矩阵 
        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden # affine transformation
        x = x.view(-1, self.k, self.k)
        return x

# PointNet编码器
class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()

        self.stn = STN3d(channel) # STN3d: T-Net 3*3 transform
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64) # STNkd: T-Net 64*64 transform

    def forward(self, x):
        B, D, N = x.size() # batchsize，3（xyz坐标）或6（xyz坐标+法向量），1024(一个物体所取的点的数目）
        trans = self.stn(x) # STN3d T-Net
        x = x.transpose(2, 1) # 交换一个tensor的两个维度
        if D >3 :
            x, feature = x.split(3,dim=2)
        # 对输入的点云进行输入转换(input transform)    
        # input transform: 计算两个tensor的矩阵乘法
        # bmm是两个三维张量相乘, 两个输入tensor维度是(b×n×m)和(b×m×p), 
        # 第一维b代表batch size，输出为(b×n×p)
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x,feature],dim=2) 
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x))) # MLP

        if self.feature_transform:
            trans_feat = self.fstn(x) # STNkd T-Net
            x = x.transpose(2, 1)
            # 对输入的点云进行特征转换(feature transform)
            # feature transform: 计算两个tensor的矩阵乘法
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x # 局部特征
        x = F.relu(self.bn2(self.conv2(x))) # MLP
        x = self.bn3(self.conv3(x)) # MLP
        x = torch.max(x, 2, keepdim=True)[0] # 最大池化得到全局特征
        x = x.view(-1, 1024) # 展平
        if self.global_feat: # 需要返回的是否是全局特征?
            return x, trans, trans_feat # 返回全局特征
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            # 返回局部特征与全局特征的拼接
            return torch.cat([x, pointfeat], 1), trans, trans_feat

# 对特征转换矩阵做正则化：
# constrain the feature transformation matrix to be close to orthogonal matrix
def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :] # torch.eye(n, m=None, out=None) 返回一个2维张量，对角线位置全1，其它位置全0
    if trans.is_cuda:
        I = I.cuda()
        
    # 正则化损失函数
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1) - I), dim=(1, 2))) 
    return loss
