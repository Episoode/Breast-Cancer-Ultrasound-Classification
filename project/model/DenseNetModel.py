import torch
from torch import nn
from torchvision.models.densenet import densenet121

# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

# 逐任务解耦模块 
class TaskSpecificDecoupling(nn.Module):
    def __init__(self, input_dim, shared_ratio=0.5, num_tasks=4):
        super(TaskSpecificDecoupling, self).__init__()
        shared_dim = int(input_dim * shared_ratio)
        task_dim = 512  # 调整每个任务的输出维度以匹配 classifier 的输入
        self.shared_proj = nn.Linear(input_dim, shared_dim)
        self.task_proj = nn.ModuleList([nn.Linear(input_dim, task_dim - shared_dim) for _ in range(num_tasks)])

    def forward(self, x):
        shared_features = self.shared_proj(x)
        task_outputs = [proj(x) for proj in self.task_proj]
        return [torch.cat([shared_features, task_out], dim=1) for task_out in task_outputs]

# 带有通道注意力和空间注意力的DenseBlock   
class DenseBlockWithAtten(nn.Module):
    def __init__(self, denseblock, in_channels, ratio=16, spatial=True):
        super(DenseBlockWithAtten, self).__init__()
        self.denseblock = denseblock
        self.channel_attention = ChannelAttention(in_channels, ratio)
        self.spatial_attention = SpatialAttention() if spatial else None

    def forward(self, x):
        x = self.denseblock(x)
        x = self.channel_attention(x) * x
        if self.spatial_attention:
            x = self.spatial_attention(x) * x
        return x

# 基于densenet121的改进网络
class MyDenseNet(nn.Module):
    def __init__(self, num_tasks=4, dp1=0.5, dp2=0.03):
        super(MyDenseNet, self).__init__()
        self.densenet = densenet121()
        channels = [256, 512, 1024, 1024]

        # 修改 denseblock，添加 Dropout 层和 Attention
        for i in range(1, 5):
            denseblock = getattr(self.densenet.features, f'denseblock{i}')
            for j, layer in denseblock.named_children():
                denselayer = getattr(denseblock, j)
                new_layer = nn.Sequential(
                    denselayer,
                    nn.Dropout(p=dp2)
                )
                setattr(denseblock, j, new_layer)

            in_channels = channels[i - 1]
            new_denseblock = DenseBlockWithAtten(denseblock, in_channels, ratio=16 * (i+1) // 2)
            setattr(self.densenet.features, f'denseblock{i}', new_denseblock)

        num_features = self.densenet.classifier.in_features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 改进的分类器
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=num_features),
            nn.Linear(in_features=num_features, out_features=num_features // 2),
            nn.ReLU(),
            nn.Dropout(p=dp1),
            nn.Linear(in_features=num_features // 2, out_features=num_features // 4),
            nn.GELU()
        )
        
        # 添加解耦模块，将特征分为共享和任务特定部分
        self.decoupling = TaskSpecificDecoupling(num_features // 4, shared_ratio=0.5, num_tasks=num_tasks)
        
        # 多任务分类头
        self.task_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_features // 2, num_features // 8),
                nn.ReLU(),
                nn.Linear(num_features // 8, 2)
            ) for _ in range(num_tasks)
        ])

    def forward(self, x):
        x = self.densenet.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        # 通过解耦模块
        task_features = self.decoupling(x)

        outputs = []
        for feat, classifier in zip(task_features, self.task_classifiers):
            outputs.append(classifier(feat))

        return torch.stack(outputs, dim=1)
