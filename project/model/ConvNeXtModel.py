import torch
import torchvision
from torch import nn
from torchvision.models.convnext import CNBlock

# 修改ConvNeXtBlock，增加dropout和随机深度
class ConvNeXtBlockWithDropout(nn.Module):
    def __init__(self, block, dp_rate=0.3, sd_rate=0.5):
        super(ConvNeXtBlockWithDropout, self).__init__()
        self.block = block
        if sd_rate > 0.0:
            self.block.stochastic_depth.p = sd_rate
        self.dropout = nn.Dropout(p=dp_rate)

    def forward(self, x):
        x = self.block(x)
        x = self.dropout(x)
        return x

# 基于convnext_tiny模型，对CNBlock进行改进
class MyConvNeXt(nn.Module):
    def __init__(self, num_classes=6, dropout_rate=0.3, sd_rate=0.3):
        super(MyConvNeXt, self).__init__()
        self.convnext_tiny = torchvision.models.convnext_tiny()

        total_num = 18
        CNB_idx = 1

        for stage in self.convnext_tiny.features:
            if isinstance(stage, nn.Sequential):
                for i, block in enumerate(stage):
                    if isinstance(block,CNBlock):
                        cur_sd_rate = sd_rate * (CNB_idx-1)/(total_num-1)
                        CNB_idx += 1
                    else :
                        cur_sd_rate = -1
                    stage[i] = ConvNeXtBlockWithDropout(block, dropout_rate,cur_sd_rate)

        num_features = self.convnext_tiny.classifier[-1].in_features
        self.convnext_tiny.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=num_features),
            nn.Linear(in_features=num_features, out_features=num_features // 2),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=num_features // 2, out_features=num_classes)
        )

    def forward(self, x):
        x = self.convnext_tiny.features(x)
        x = self.convnext_tiny.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.convnext_tiny.classifier(x)
        return x
    