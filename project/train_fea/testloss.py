import torch
from torch import nn


class F1Loss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(F1Loss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        # 对每个任务的输出进行 Softmax 归一化
        y_pred = torch.softmax(y_pred, dim=-1) 
        total_f1_loss = 0

        for i in range(y_pred.size(1)):  
            task_pred = y_pred[:, i, :]  # (batch, 2)
            task_true = y_true[:, i]  # (batch,)
            # 获取类别1的预测概率
            p0 = task_pred[:, 0]
            p1 = task_pred[:, 1]
            
            TP = (p1 * task_true).sum()
            FP = (p1 * (1 - task_true)).sum()
            FN = (p0 * task_true).sum()

            # 为了避免数值不稳定，确保对数操作的输入不为零
            TP = torch.clamp(TP, min=self.smooth)
            FP = torch.clamp(FP, min=self.smooth)
            FN = torch.clamp(FN, min=self.smooth)

            # 计算 Precision 和 Recall
            precision = TP / (TP + FP + self.smooth)
            recall = TP / (TP + FN + self.smooth)

            # 计算 F1 
            f1_score = 2 * (precision * recall) / (precision + recall + self.smooth)

            # F1 损失累加
            total_f1_loss += (1 - f1_score)

        return total_f1_loss


# if __name__ == "__main__":
#     for i in range(10):
#         # 模拟模型输出 (batch, num_task, 2) 形状，要求每个任务有两个类别的输出
#         y_pred = torch.randn(10, 4, 2, requires_grad=True)  # 假设 num_task=4，即4个二分类任务
#
#         # 随机生成标签，形状为 (batch, num_task)
#         y_true = torch.randint(0, 2, (10, 4))  # 每个任务对应一个标签，值为0或1
#
#         # 初始化 F1 损失函数
#         loss_fn = F1Loss()
#
#         loss = loss_fn(y_pred, y_true)
#         print("True:\n", y_true)
#         print(f'Loss before backward: {loss.item()}')
#
#         loss.backward()
#         print(f'Gradient of y_pred: {y_pred.grad}')
#
#         y_pred.grad.zero_()
