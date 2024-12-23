import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from tqdm import tqdm
from PIL import Image
import os

# 加上batch normalization效果不错，对lr的选取增加了容错
# lr = 0.0006, step_size = 5, gamma = 0.2   0.58  0.56  感觉学习率偏小？  MDvsFA竟然更大一些
# lr = 0.002, step_size = 5, gamma = 0.2   0.57  0.63  效果不错，可以再大一些
# lr = 0.01, step_size = 5, gamma = 0.2   0.93  0.60  0.64   学习率再大一点试试
# lr = 0.012, step_size = 5, gamma = 0.2
# lr = 0.015, step_size = 5, gamma = 0.2   0.53 + 0.62
# lr = 0.02, step_size = 5, gamma = 0.2   不是很稳定  有些过拟合  0.58 0.67

# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
k = 8

class ScaleFusionModule(nn.Module):
    def __init__(self, out_channels, out_size):
        super(ScaleFusionModule, self).__init__()

        # 多尺度特征融合子模块
        self.conv_low = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv_mid = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv_high = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.pool_mid = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool_high = nn.MaxPool2d(2)
        self.BN1 = nn.BatchNorm2d(out_channels)
        self.BN2 = nn.BatchNorm2d(out_channels)
        self.BN3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.attention_BN1 = nn.BatchNorm2d(3 * out_channels)
        self.attention_BN2 = nn.BatchNorm2d(3 * out_channels)
        self.attention_BN3 = nn.BatchNorm2d(3 * out_channels)

        # 尺度注意力生成子模块
        self.attention_conv1 = nn.Conv2d(out_channels * 3, out_channels * 3, kernel_size=3, padding=1)
        self.attention_conv2 = nn.Conv2d(out_channels * 3, out_channels * 3, kernel_size=3, padding=1)
        self.attention_conv3 = nn.Conv2d(out_channels * 3, out_channels * 3, kernel_size=3, padding=1)
        self.softmax = nn.Softmax(dim=1)

        self.out_size = out_size  # 输出标准化尺寸

    def forward(self, low, mid, high):
        # 多尺度特征融合子模块
        low = self.conv_low(low)
        mid = self.conv_mid(mid)
        high = self.conv_high(high)
        # mid = self.conv_mid(nn.functional.interpolate(mid, size=low.shape[2:], mode='bilinear', align_corners=False))
        # high = self.conv_high(nn.functional.interpolate(high, size=low.shape[2:], mode='bilinear', align_corners=False))

        # low = self.relu(self.BN(low))
        # mid = self.relu(self.BN(mid))
        # high = self.relu(self.BN(high))
        low = self.BN1(low)
        mid = self.BN2(mid)
        high = self.BN3(high)

        low = self.relu(low)
        mid = self.relu(mid)
        high = self.relu(high)

        mid = nn.functional.interpolate(mid, size=low.shape[2:], mode='bilinear', align_corners=False)
        high = nn.functional.interpolate(high, size=low.shape[2:], mode='bilinear', align_corners=False)
        # mid = self.pool_mid(mid)
        # high = self.pool_high(high)

        # 合并特征
        merged = torch.cat((low, mid, high), dim=1)

        # 尺度注意力生成子模块
        attention_weights = self.relu(self.attention_BN1(self.attention_conv1(merged)))
        attention_weights = self.relu(self.attention_BN2(self.attention_conv2(attention_weights)))
        attention_weights = self.softmax(self.attention_BN3(self.attention_conv3(attention_weights)))  # 生成尺度注意力权重

        attention = nn.functional.adaptive_avg_pool2d(attention_weights, (1, 1))  # 输出形状 (B, channels, 1, 1)

        attention_low, attention_mid, attention_high = torch.split(attention, low.shape[1], dim=1)

        # 权重作用到不同尺度特征
        low = attention_low * low
        mid = attention_mid * mid
        high = attention_high * high

        # 最终输出
        output = low + mid + high
        return output

# ResNet Block
class ResNet(nn.Module):
    def __init__(self, in_channels):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 2 * in_channels, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(2 * in_channels, 2 * in_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(2 * in_channels, 2 * in_channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(2 * in_channels, 2 * in_channels, kernel_size=3, stride=1, padding=1)

        self.upsample = nn.Conv2d(in_channels, 2 * in_channels, kernel_size=1, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(2 * in_channels)
        self.bn2 = nn.BatchNorm2d(2 * in_channels)

    def forward(self, x):
        x_1 = self.relu(self.conv1(x))
        x_sample = self.upsample(x)
        x_2 = self.relu(self.bn1(self.conv2(x_1) + x_sample))

        x_3 = self.relu(self.conv3(x_2))
        x_4 = self.conv4(x_3)

        # return x_2
        return self.relu(self.bn2(x_4 + x_2))

# CFM模块
class ChannelFusionModule(nn.Module):
    def __init__(self, in_channels):
        super(ChannelFusionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # attention = self.sigmoid(self.conv1(x))
        # return x * attention
        x = self.conv1(x)
        return x

# WTL模块
class WTL(nn.Module):
    def __init__(self, in_channels):
        super(WTL, self).__init__()
        self.conv = nn.Conv2d(in_channels, 2 * in_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool(x)
        return x

# BPR-Net模型
class BPRNet(nn.Module):
    def __init__(self):
        super(BPRNet, self).__init__()

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        #定义Resnet
        self.encoder2 = ResNet(2)
        self.encoder3 = ResNet(4)
        self.encoder4 = ResNet(8)
        self.encoder5 = ResNet(16)

        # 编码器卷积层
        self.encoder_conv1 = nn.Conv2d(1, 2, kernel_size=7, padding=3)
        # self.encoder_conv2 = nn.Conv2d(2, 4, kernel_size=5, padding=2)
        # self.encoder_conv3 = nn.Conv2d(4, 8, kernel_size=5, padding=2)
        # self.encoder_conv4 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        # self.encoder_conv5 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # 卷积层将所有特征都变为k通道
        self.low_conv1_1 = nn.Conv2d(2, 4, kernel_size=5, padding=2)
        self.low_conv1_2 = nn.Conv2d(4, k, kernel_size=5, padding=2)
        self.low_conv1_3 = nn.Conv2d(k, k, kernel_size=5, padding=2)
        self.low_conv1_4 = nn.Conv2d(k, k, kernel_size=3, padding=1)
        self.low_conv1_5 = nn.Conv2d(k, k, kernel_size=3, padding=1)
        self.mid_conv1_1 = nn.Conv2d(2, 4, kernel_size=5, padding=2)
        self.mid_conv1_2 = nn.Conv2d(4, k, kernel_size=5, padding=2)
        self.mid_conv1_3 = nn.Conv2d(k, k, kernel_size=5, padding=2)
        self.mid_conv1_4 = nn.Conv2d(k, k, kernel_size=3, padding=1)
        self.mid_conv1_5 = nn.Conv2d(k, k, kernel_size=3, padding=1)
        self.high_conv1_1 = nn.Conv2d(2, 4, kernel_size=5, padding=2)
        self.high_conv1_2 = nn.Conv2d(4, k, kernel_size=5, padding=2)
        self.high_conv1_3 = nn.Conv2d(k, k, kernel_size=5, padding=2)
        self.high_conv1_4 = nn.Conv2d(k, k, kernel_size=3, padding=1)
        self.high_conv1_5 = nn.Conv2d(k, k, kernel_size=3, padding=1)

        self.low_conv2_1 = nn.Conv2d(4, k, kernel_size=3, padding=1)
        self.low_conv2_2 = nn.Conv2d(k, k, kernel_size=3, padding=1)
        # self.low_conv2_3 = nn.Conv2d(k, k, kernel_size=3, padding=1)
        # self.low_conv2_4 = nn.Conv2d(k, k, kernel_size=3, padding=1)
        self.mid_conv2_1 = nn.Conv2d(4, k, kernel_size=3, padding=1)
        self.mid_conv2_2 = nn.Conv2d(k, k, kernel_size=3, padding=1)
        # self.mid_conv2_3 = nn.Conv2d(k, k, kernel_size=3, padding=1)
        # self.mid_conv2_4 = nn.Conv2d(k, k, kernel_size=3, padding=1)
        self.high_conv2_1 = nn.Conv2d(4, k, kernel_size=3, padding=1)
        self.high_conv2_2 = nn.Conv2d(k, k, kernel_size=3, padding=1)
        # self.high_conv2_3 = nn.Conv2d(k, k, kernel_size=3, padding=1)
        # self.high_conv2_4 = nn.Conv2d(k, k, kernel_size=3, padding=1)

        self.low_conv3_1 = nn.Conv2d(8, k, kernel_size=3, padding=1)
        # self.low_conv3_2 = nn.Conv2d(8, k, kernel_size=3, padding=1)
        # self.low_conv3_3 = nn.Conv2d(8, k, kernel_size=3, padding=1)
        self.mid_conv3_1 = nn.Conv2d(8, k, kernel_size=3, padding=1)
        # self.mid_conv3_2 = nn.Conv2d(8, k, kernel_size=3, padding=1)
        # self.mid_conv3_3 = nn.Conv2d(8, k, kernel_size=3, padding=1)
        self.high_conv3_1 = nn.Conv2d(8, k, kernel_size=3, padding=1)
        # self.high_conv3_2 = nn.Conv2d(8, k, kernel_size=3, padding=1)
        # self.high_conv3_3 = nn.Conv2d(8, k, kernel_size=3, padding=1)

        # self.low_conv4_1 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.low_conv4_2 = nn.Conv2d(16, k, kernel_size=3, padding=1)
        # self.mid_conv4_1 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.mid_conv4_2 = nn.Conv2d(16, k, kernel_size=3, padding=1)
        # self.high_conv4_1 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.high_conv4_2 = nn.Conv2d(16, k, kernel_size=3, padding=1)

        self.low_conv5_1 = nn.Conv2d(32, k, kernel_size=3, padding=1)
        self.mid_conv5_1 = nn.Conv2d(32, k, kernel_size=3, padding=1)
        self.high_conv5_1 = nn.Conv2d(32, k, kernel_size=3, padding=1)

        # SFM模块
        self.sfm_1 = ScaleFusionModule(k, 64)
        self.sfm_2 = ScaleFusionModule(k, 32)
        self.sfm_3 = ScaleFusionModule(k, 16)
        self.sfm_4 = ScaleFusionModule(k, 8)
        self.sfm_5 = ScaleFusionModule(k, 4)

        # WTL模块，参数暂时没有用
        self.wtl_2 = WTL(16)
        self.wtl_3 = WTL(8)
        self.wtl_4 = WTL(4)
        self.wtl_5 = WTL(2)

        # CFM模块，参数暂时也没有用
        self.CFM_1 = ChannelFusionModule(k)
        self.CFM_2 = ChannelFusionModule(k)
        self.CFM_3 = ChannelFusionModule(k)
        self.CFM_4 = ChannelFusionModule(k)
        self.CFM_5 = ChannelFusionModule(k)

        self.final_conv = nn.Conv2d(k, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.bnlow1_1 = nn.BatchNorm2d(4)
        self.bnlow1_2 = nn.BatchNorm2d(8)
        self.bnlow1_3 = nn.BatchNorm2d(8)
        self.bnlow1_4 = nn.BatchNorm2d(8)
        self.bnlow1_5 = nn.BatchNorm2d(8)

        self.bnmid1_1 = nn.BatchNorm2d(4)
        self.bnmid1_2 = nn.BatchNorm2d(8)
        self.bnmid1_3 = nn.BatchNorm2d(8)
        self.bnmid1_4 = nn.BatchNorm2d(8)
        self.bnmid1_5 = nn.BatchNorm2d(8)

        self.bnhigh1_1 = nn.BatchNorm2d(4)
        self.bnhigh1_2 = nn.BatchNorm2d(8)
        self.bnhigh1_3 = nn.BatchNorm2d(8)
        self.bnhigh1_4 = nn.BatchNorm2d(8)
        self.bnhigh1_5 = nn.BatchNorm2d(8)

        self.bnlow2_1 = nn.BatchNorm2d(8)
        self.bnlow2_2 = nn.BatchNorm2d(8)

        self.bnmid2_1 = nn.BatchNorm2d(8)
        self.bnmid2_2 = nn.BatchNorm2d(8)

        self.bnhigh2_1 = nn.BatchNorm2d(8)
        self.bnhigh2_2 = nn.BatchNorm2d(8)

        self.bnlow3_1 = nn.BatchNorm2d(8)
        self.bnmid3_1 = nn.BatchNorm2d(8)
        self.bnhigh3_1 = nn.BatchNorm2d(8)

        self.bnlow4_1 = nn.BatchNorm2d(8)
        self.bnmid4_1 = nn.BatchNorm2d(8)
        self.bnhigh4_1 = nn.BatchNorm2d(8)

        self.bnlow5_1 = nn.BatchNorm2d(8)
        self.bnmid5_1 = nn.BatchNorm2d(8)
        self.bnhigh5_1 = nn.BatchNorm2d(8)

        self.final_bn = nn.BatchNorm2d(1)

    def forward(self, x):
        x_mid = nn.functional.interpolate(x, scale_factor=1.5)
        x_high = nn.functional.interpolate(x, scale_factor=2)

        # 编码器生成特征
        low_feat_1 = self.relu(self.encoder_conv1(x))
        low_feat_2 = self.encoder2(low_feat_1)
        low_feat_3 = self.encoder3(low_feat_2)
        low_feat_4 = self.encoder4(low_feat_3)
        low_feat_5 = self.encoder5(low_feat_4)

        mid_feat_1 = self.relu(self.encoder_conv1(x_mid))
        mid_feat_2 = self.encoder2(mid_feat_1)
        mid_feat_3 = self.encoder3(mid_feat_2)
        mid_feat_4 = self.encoder4(mid_feat_3)
        mid_feat_5 = self.encoder5(mid_feat_4)

        high_feat_1 = self.relu(self.encoder_conv1(x_high))
        high_feat_2 = self.encoder2(high_feat_1)
        high_feat_3 = self.encoder3(high_feat_2)
        high_feat_4 = self.encoder4(high_feat_3)
        high_feat_5 = self.encoder5(high_feat_4)

        low_feat_1 = self.relu(self.bnlow1_1(self.low_conv1_1(low_feat_1)))
        low_feat_1 = self.relu(self.bnlow1_2(self.low_conv1_2(low_feat_1)))
        low_feat_1 = self.relu(self.bnlow1_3(self.low_conv1_3(low_feat_1)))
        low_feat_1 = self.relu(self.bnlow1_4(self.low_conv1_4(low_feat_1)))
        low_feat_1 = self.relu(self.bnlow1_5(self.low_conv1_5(low_feat_1)))
        mid_feat_1 = self.relu(self.bnmid1_1(self.mid_conv1_1(mid_feat_1)))
        mid_feat_1 = self.relu(self.bnmid1_2(self.mid_conv1_2(mid_feat_1)))
        mid_feat_1 = self.relu(self.bnmid1_3(self.mid_conv1_3(mid_feat_1)))
        mid_feat_1 = self.relu(self.bnmid1_4(self.mid_conv1_4(mid_feat_1)))
        mid_feat_1 = self.relu(self.bnmid1_5(self.mid_conv1_5(mid_feat_1)))
        high_feat_1 = self.relu(self.bnhigh1_1(self.high_conv1_1(high_feat_1)))
        high_feat_1 = self.relu(self.bnhigh1_2(self.high_conv1_2(high_feat_1)))
        high_feat_1 = self.relu(self.bnhigh1_3(self.high_conv1_3(high_feat_1)))
        high_feat_1 = self.relu(self.bnhigh1_4(self.high_conv1_4(high_feat_1)))
        high_feat_1 = self.relu(self.bnhigh1_5(self.high_conv1_5(high_feat_1)))

        low_feat_2 = self.relu(self.bnlow2_1(self.low_conv2_1(low_feat_2)))
        low_feat_2 = self.relu(self.bnlow2_2(self.low_conv2_2(low_feat_2)))
        mid_feat_2 = self.relu(self.bnmid2_1(self.mid_conv2_1(mid_feat_2)))
        mid_feat_2 = self.relu(self.bnmid2_2(self.mid_conv2_2(mid_feat_2)))
        high_feat_2 = self.relu(self.bnhigh2_1(self.high_conv2_1(high_feat_2)))
        high_feat_2 = self.relu(self.bnhigh2_2(self.high_conv2_2(high_feat_2)))

        low_feat_3 = self.relu(self.bnlow3_1(self.low_conv3_1(low_feat_3)))
        mid_feat_3 = self.relu(self.bnmid3_1(self.mid_conv3_1(mid_feat_3)))
        high_feat_3 = self.relu(self.bnhigh3_1(self.high_conv3_1(high_feat_3)))

        low_feat_4 = self.relu(self.bnlow4_1(self.low_conv4_2(low_feat_4)))
        mid_feat_4 = self.relu(self.bnmid4_1(self.mid_conv4_2(mid_feat_4)))
        high_feat_4 = self.relu(self.bnhigh4_1(self.high_conv4_2(high_feat_4)))

        low_feat_5 = self.relu(self.bnlow5_1(self.low_conv5_1(low_feat_5)))
        mid_feat_5 = self.relu(self.bnmid5_1(self.mid_conv5_1(mid_feat_5)))
        high_feat_5 = self.relu(self.bnhigh5_1(self.high_conv5_1(high_feat_5)))

        # SFM
        f1 = self.sfm_5(low_feat_1, mid_feat_1, high_feat_1)
        f2 = self.sfm_4(low_feat_2, mid_feat_2, high_feat_2)
        f3 = self.sfm_3(low_feat_3, mid_feat_3, high_feat_3)
        f4 = self.sfm_2(low_feat_4, mid_feat_4, high_feat_4)
        f5 = self.sfm_1(low_feat_5, mid_feat_5, high_feat_5)

        # WTL
        # wtl_5 = self.wtl_5(f4)
        # wtl_4 = self.wtl_4(f3)
        # wtl_3 = self.wtl_3(f2)
        # wtl_2 = self.wtl_2(f1)

        # CFM
        cfm_5 = self.CFM_5(f5)
        c5 = cfm_5 + f5

        cfm_4 = self.CFM_4(f4 + self.upsample(c5))
        c4 = cfm_4 + f4

        cfm_3 = self.CFM_3(f3 + self.upsample(c4))
        c3 = cfm_3 + f3

        cfm_2 = self.CFM_2(f2 + self.upsample(c3))
        c2 = cfm_2 + f2

        cfm_1 = self.CFM_1(f1 + self.upsample(c2))
        c1 = cfm_1 + f1

        output = self.sigmoid(self.final_bn(self.final_conv(c1)))
        return output

# Dice损失函数
def dice_loss(pred, target, smooth=1e-5):
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

# 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('L')
        mask = Image.open(self.mask_paths[idx]).convert('L')
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask

def iou_loss(pred_boxes, target_boxes):
    """
    计算候选框的 IoU 损失。
    """
    # 计算交集
    inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # 计算并集
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
    union_area = pred_area + target_area - inter_area

    # IoU 和损失
    iou = inter_area / torch.clamp(union_area, min=1e-5)
    return 1 - iou.mean()

# 训练函数
def train_model(model, train_loader, val_loader, num_epochs):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
    criterion = nn.BCELoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images, masks = images.to(device).float(), masks.to(device).float()
            optimizer.zero_grad()
            outputs = model(images)

            loss = 10 * criterion(outputs, masks) + 0.2 * dice_loss(outputs, masks)
            # loss = 10 * criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 更新学习率
        scheduler.step()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

        # 验证
        model.eval()

        all_preds = []
        all_labels = []

        for img_path, mask_path in zip(val_images, val_masks):
            # for img_path, mask_path in zip(train_images, train_masks):
            # 加载图片和掩码
            image, original_size = preprocess_image(img_path)
            mask = Image.open(mask_path).convert('L')
            mask = transforms.ToTensor()(mask)  # 转为张量

            # 执行模型预测
            image = image.unsqueeze(0).to(device)  # 增加 batch 维度
            with torch.no_grad():
                output = model(image)  # 模型输出 (128x128)

            # 将输出调整回原始尺寸
            output_resized = resize_to_original(output, original_size).cpu()
            preds_binary = (output_resized > 0.5).int()  # 二值化预测结果

            # 收集预测和真实值
            all_preds.append(preds_binary.flatten())
            all_labels.append(mask.flatten())

        # 拼接所有预测和真实值
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        # 计算总的 F1 score
        f1_score_test = calculateF1Measure(all_preds, all_labels, thre=0.5)
        # f1_score_test = f1_score(all_labels, all_preds)
        print(f"F1 Score in SIRST: {f1_score_test:.4f}")

        all_preds = []
        all_labels = []

        for img_path, mask_path in zip(val_images1, val_masks1):
            # for img_path, mask_path in zip(train_images, train_masks):
            # 加载图片和掩码
            image, original_size = preprocess_image(img_path)
            mask = Image.open(mask_path).convert('L')
            mask = transforms.ToTensor()(mask)  # 转为张量

            # 执行模型预测
            image = image.unsqueeze(0).to(device)  # 增加 batch 维度
            with torch.no_grad():
                output = model(image)  # 模型输出 (128x128)

            # 将输出调整回原始尺寸
            output_resized = resize_to_original(output, original_size).cpu()
            preds_binary = (output_resized > 0.5).int()  # 二值化预测结果

            # 收集预测和真实值
            all_preds.append(preds_binary.flatten())
            all_labels.append(mask.flatten())

        # 拼接所有预测和真实值
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        # 计算总的 F1 score
        f1_score_test = calculateF1Measure(all_preds, all_labels, thre=0.5)
        # f1_score_test = f1_score(all_labels, all_preds)
        print(f"F1 Score in FDvsMA: {f1_score_test:.4f}")

def calculateF1Measure(output_image, gt_image, thre):
    output_image = np.squeeze(output_image)
    gt_image = np.squeeze(gt_image)
    # out_bin = output_image > thre
    # gt_bin = gt_image > thre
    out_bin = output_image
    gt_bin = gt_image
    recall = np.sum(gt_bin * out_bin) / np.maximum(1, np.sum(gt_bin))
    prec = np.sum(gt_bin * out_bin) / np.maximum(1, np.sum(out_bin))
    F1 = 2 * recall * prec / np.maximum(0.001, recall+prec)
    # F1 = 2 * recall * prec / (recall + prec)

    print(f"recall Score: {recall:.4f}")

    print(f"prec Score: {prec:.4f}")
    return F1

# 可视化预测结果
def visualize_predictions(model, val_loader):
    model.to(device)
    model.eval()
    images, masks = next(iter(val_loader))
    images, masks = images.to(device).float(), masks.to(device).float()

    with torch.no_grad():
        preds = model(images)
        preds = nn.functional.interpolate(preds, size=masks.shape[2:], mode='bilinear', align_corners=False)
        preds = (preds > 0.5).cpu().numpy()

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(images[0].cpu().numpy().squeeze(), cmap='gray')
    plt.title('Input Image')

    plt.subplot(1, 3, 2)
    plt.imshow(masks[0].cpu().numpy().squeeze(), cmap='gray')
    plt.title('Ground Truth')

    plt.subplot(1, 3, 3)
    plt.imshow(preds[0].squeeze(), cmap='gray')
    plt.title('Predicted Mask')

    plt.show()

def evaluate_model(model, val_loader):
    """
    在验证集上评估模型并计算 F1 分数。
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device).float(), masks.to(device).float()
            outputs = model(images)

            # 二值化预测
            preds = (outputs > 0.5).int()

            # 收集预测和真实标签
            all_preds.append(preds)
            all_labels.append(masks.int())

    # 拼接所有预测和标签
    all_preds = torch.cat(all_preds).view(-1).cpu().numpy()
    all_labels = torch.cat(all_labels).view(-1).cpu().numpy()

    # 计算 F1 分数
    f1 = f1_score(all_labels, all_preds)
    print(f"Validation F1 Score: {f1:.4f}")
    return f1

def save_predictions(model, data_loader, save_dir, num_images=100):
    """
    保存模型预测结果到指定路径。
    """
    model.eval()
    count = 0

    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device).float()
            outputs = model(images)

            # 将预测结果二值化
            preds = (outputs > 0.5).float()

            # 遍历当前批次的所有图片
            for i in range(images.size(0)):
                if count >= num_images:
                    return  # 保存指定数量后退出

                # 保存预测结果
                save_path = os.path.join(save_dir, f'prediction_{count + 1}.png')
                save_image(preds[i], save_path)
                count += 1

def preprocess_image(image_path):
    """加载图片并调整为 128x128 大小"""
    image = Image.open(image_path).convert('L')  # 灰度化
    original_size = image.size  # 保存原始尺寸 (width, height)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    image = transform(image)
    return image, original_size

def resize_to_original(output, original_size):
    height, width = original_size[1], original_size[0]  # 转换为 (height, width)
    return torch.nn.functional.interpolate(output, size=(height, width), mode='bilinear', align_corners=False)

# 数据路径
image_dir = 'C:/Users/admin/Desktop/视觉认知工程课设/Dataset/数据集/训练集/image'
mask_dir = 'C:/Users/admin/Desktop/视觉认知工程课设/Dataset/数据集/训练集/mask'
train_images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]
train_masks = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')]

image_dir1 = 'C:/Users/admin/Desktop/视觉认知工程课设/Dataset/数据集/测试集/SIRST/image'
mask_dir1 = 'C:/Users/admin/Desktop/视觉认知工程课设/Dataset/数据集/测试集/SIRST/mask'
image_dir2 = 'C:/Users/admin/Desktop/视觉认知工程课设/Dataset/数据集/测试集/MDvsFA/image'
mask_dir2 = 'C:/Users/admin/Desktop/视觉认知工程课设/Dataset/数据集/测试集/MDvsFA/mask'

train_images1 = 'C:/Users/admin/Desktop/视觉认知工程课设/Dataset/数据集/测试集/TEST/image'
train_masks1 = 'C:/Users/admin/Desktop/视觉认知工程课设/Dataset/数据集/测试集/TEST/mask'

val_images = [os.path.join(image_dir1, f) for f in os.listdir(image_dir1) if f.endswith('.png')]
val_masks = [os.path.join(mask_dir1, f) for f in os.listdir(mask_dir1) if f.endswith('.png')]
val_images1 = [os.path.join(image_dir2, f) for f in os.listdir(image_dir2) if f.endswith('.png')]
val_masks1 = [os.path.join(mask_dir2, f) for f in os.listdir(mask_dir2) if f.endswith('.png')]

save_dir = 'C:/Users/admin/Desktop/视觉认知工程课设/code/RBP-Net/result/MDvsFA'
os.makedirs(save_dir, exist_ok=True)

# 模型保存路径
model_dir = 'C:/Users/admin/Desktop/视觉认知工程课设/code/RBP-Net/result'
os.makedirs(model_dir, exist_ok=True)  # 如果路径不存在，自动创建
model_path = os.path.join(model_dir, "model_weights.pth")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

train_dataset = CustomDataset(train_images, train_masks, transform)
val_dataset = CustomDataset(val_images, val_masks, transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

if __name__ == "__main__":

    # 训练模型
    model = BPRNet().to(device)
    train_model(model, train_loader, val_loader, 20)

    # model.eval()
    # print("Evaluating model on validation set...")
    # f1_score_val = evaluate_model(model, val_loader)

    # 保存网络参数，便于测试
    torch.save(model.state_dict(), model_path)
    print("Model weights saved to 'model_weights.pth'")

    save_predictions(model, train_loader, save_dir, num_images=10)
    # visualize_predictions(model, val_loader)
