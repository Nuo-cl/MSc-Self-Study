import torch
import torch.nn as nn

class InceptionModule(nn.Module):
    """
    Inception模块（带降维）
    
    参数:
        in_channels: 输入通道数
        n1x1: 1x1卷积输出通道数
        n3x3_reduce: 3x3卷积前的1x1降维输出通道数
        n3x3: 3x3卷积输出通道数
        n5x5_reduce: 5x5卷积前的1x1降维输出通道数
        n5x5: 5x5卷积输出通道数
        pool_proj: 池化后的1x1投影输出通道数
    """
    def __init__(self, in_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
        super(InceptionModule, self).__init__()
        
        # 第一条路径: 1x1卷积
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, n1x1, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        # 第二条路径: 1x1降维 -> 3x3卷积
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, n3x3_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n3x3_reduce, n3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 第三条路径: 1x1降维 -> 5x5卷积
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, n5x5_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5_reduce, n5x5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        
        # 第四条路径: 3x3最大池化 -> 1x1投影
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 并行计算四条路径
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        branch3_out = self.branch3(x)
        branch4_out = self.branch4(x)
        
        # 在通道维度拼接
        outputs = torch.cat([branch1_out, branch2_out, branch3_out, branch4_out], dim=1)
        return outputs


class AuxiliaryClassifier(nn.Module):
    """
    辅助分类器
    用于在网络中间层添加额外的监督信号
    """
    def __init__(self, in_channels, num_classes):
        super(AuxiliaryClassifier, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Linear(2048, 1024)  # 对于14x14输入，池化后为4x4，128*4*4=2048
        self.fc2 = nn.Linear(1024, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.7)
    
    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GoogLeNet(nn.Module):
    """
    GoogLeNet (Inception v1)
    22层深度网络（不包含池化层）
    输入尺寸: 224x224x3
    """
    def __init__(self, num_classes=1000, aux_classifiers=True):
        super(GoogLeNet, self).__init__()
        self.aux_classifiers = aux_classifiers
        
        # 初始层（传统卷积）
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # -> 56x56x64
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # -> 28x28x192
        )
        
        # Inception模块层
        # Inception 3a, 3b (28x28)
        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)  # -> 28x28x256
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)  # -> 28x28x480
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # -> 14x14x480
        
        # Inception 4a, 4b, 4c, 4d, 4e (14x14)
        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)  # -> 14x14x512
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)  # -> 14x14x512
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)  # -> 14x14x512
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)  # -> 14x14x528
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)  # -> 14x14x832
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # -> 7x7x832
        
        # Inception 5a, 5b (7x7)
        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)  # -> 7x7x832
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)  # -> 7x7x1024
        
        # 辅助分类器（训练时使用）
        if self.aux_classifiers:
            self.aux1 = AuxiliaryClassifier(512, num_classes)  # 在inception4a之后
            self.aux2 = AuxiliaryClassifier(528, num_classes)  # 在inception4d之后
        
        # 分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        # 初始卷积层
        x = self.conv1(x)  # 112x112x64
        x = self.conv2(x)  # 28x28x192
        
        # Inception 3x
        x = self.inception3a(x)  # 28x28x256
        x = self.inception3b(x)  # 28x28x480
        x = self.maxpool3(x)  # 14x14x480
        
        # Inception 4x
        x = self.inception4a(x)  # 14x14x512
        
        # 第一个辅助分类器
        if self.training and self.aux_classifiers:
            aux1_out = self.aux1(x)
        
        x = self.inception4b(x)  # 14x14x512
        x = self.inception4c(x)  # 14x14x512
        x = self.inception4d(x)  # 14x14x528
        
        # 第二个辅助分类器
        if self.training and self.aux_classifiers:
            aux2_out = self.aux2(x)
        
        x = self.inception4e(x)  # 14x14x832
        x = self.maxpool4(x)  # 7x7x832
        
        # Inception 5x
        x = self.inception5a(x)  # 7x7x832
        x = self.inception5b(x)  # 7x7x1024
        
        # 全局平均池化 + 分类器
        x = self.avgpool(x)  # 1x1x1024
        x = x.view(x.size(0), -1)  # 展平
        x = self.dropout(x)
        x = self.fc(x)  # num_classes
        
        # 返回结果
        if self.training and self.aux_classifiers:
            return x, aux1_out, aux2_out
        else:
            return x

