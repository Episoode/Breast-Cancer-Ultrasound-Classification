import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
from matplotlib import pyplot as plt
from data_binary import UltrasoundDataset,test_transforms,train_transforms_new
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score
import os
from DenseNetModel import MyDenseNet
from MyF1Loss import F1Loss


def train(model, train_loader, test_loader, device, epochs=40):
    task_weights = [1.2, 1.5, 1, 1]  # 调整任务权重
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.2).to(device) # 使用标签平滑的交叉熵损失函数
    loss_fn2 = F1Loss().to(device) # 使用F1损失函数
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4) 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2) # 学习率调整器
    
    scaler = torch.cuda.amp.GradScaler()  # 混合精度缩放器
    model.train()
    train_accs, test_accs = {i: [] for i in range(1, 5)}, {i: [] for i in range(1, 5)}
    train_f1s, test_f1s = {i: [] for i in range(1, 5)}, {i: [] for i in range(1, 5)}
    max_ac = 0.0

    log_dir = './runs/dense_5' # 日志目录
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(epochs):
        train_correct = {i: 0 for i in range(1, 5)}
        train_total = {i: 0 for i in range(1, 5)}
        train_preds = {i: [] for i in range(1, 5)}
        train_labels = {i: [] for i in range(1, 5)}

        time1 = time.time()
        train_ac, test_ac, train_f, test_f = 0, 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():  # 混合精度正向传播
                outputs = model(images)
                # 加权损失
                loss = sum(task_weights[i] * loss_fn(outputs[:, i, :], labels[:, i].long()) for i in range(4))
                loss += loss_fn2(outputs, labels)

            scaler.scale(loss).backward()  # 缩放损失并反向传播
            scaler.step(optimizer)         # 优化器更新
            scaler.update()                # 更新缩放器
        
             

            for i in range(4):
                preds = torch.argmax(outputs[:, i, :], dim=1)
                train_correct[i + 1] += (preds == labels[:, i]).sum().item()
                train_total[i + 1] += labels[:, i].size(0)
                train_preds[i + 1].extend(preds.cpu().numpy())
                train_labels[i + 1].extend(labels[:, i].cpu().numpy())
        scheduler.step()   

        train_acc = {i: train_correct[i] / train_total[i] for i in range(1, 5)}
        train_f1 = {i: f1_score(train_labels[i], train_preds[i], average='binary') for i in range(1, 5)}
        test_acc, test_f1 = test(model, test_loader, device)

        for i in range(1, 5):
            train_accs[i].append(train_acc[i])
            train_f1s[i].append(train_f1[i])
            test_accs[i].append(test_acc[i])
            test_f1s[i].append(test_f1[i])

        time2 = time.time()
        epoch_time = time2 - time1
        print(f"Epoch [{epoch + 1}/{epochs}], Time: {epoch_time:.2f}s")
        for i in range(1, 5):
            print(f"Task {i}: Train Accuracy: {train_acc[i] * 100:.2f}%, Train F1: {train_f1[i] * 100:.2f}%")
            print(f"Task {i}: Test Accuracy: {test_acc[i] * 100:.2f}%, Test F1: {test_f1[i] * 100:.2f}%")
            train_ac += train_acc[i]
            test_ac += test_acc[i]
            train_f += train_f1[i]
            test_f += test_f1[i]

        print(f"Mean Accuracy: Train: {train_ac/4}, Test: {test_ac/4}")
        print(f"Mean F1: Train:{train_f/4}, Test: {test_f/4}")
        writer.add_scalar(f'Mean Test Accuracy:', test_ac/4, epoch)
        writer.add_scalar(f"Mean Test F1:", test_f/4, epoch)
        
        for i in range(1, 5):
            writer.add_scalar(f'Task_{i}/Train_Accuracy', train_acc[i], epoch)
            writer.add_scalar(f'Task_{i}/Test_Accuracy', test_acc[i], epoch)
            writer.add_scalar(f'Task_{i}/Train_F1', train_f1[i], epoch)
            writer.add_scalar(f'Task_{i}/Test_F1', test_f1[i], epoch)

        if test_ac / 4 > max_ac:
            max_ac = test_ac / 4
            if max_ac > 0.87:
                torch.save(model.state_dict(), f"./models/de_DenseNet_{max_ac:.4f}.pth")

    writer.close()
    torch.save(model.state_dict(), f"./models/DenseNet_new.pth")


def test(model, test_loader, device):
    model.eval()
    test_correct = {i: 0 for i in range(1, 5)}
    test_total = {i: 0 for i in range(1, 5)}
    test_preds = {i: [] for i in range(1, 5)}
    test_labels = {i: [] for i in range(1, 5)}

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(images)

            for i in range(4):
                preds = torch.argmax(outputs[:, i, :], dim=1)
                test_correct[i + 1] += (preds == labels[:, i]).sum().item()
                test_total[i + 1] += labels[:, i].size(0)
                test_preds[i + 1].extend(preds.cpu().numpy())
                test_labels[i + 1].extend(labels[:, i].cpu().numpy())

    test_acc = {i: test_correct[i] / test_total[i] for i in range(1, 5)}
    test_f1 = {i: f1_score(test_labels[i], test_preds[i], average='binary') for i in range(1, 5)}
    return test_acc, test_f1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
model = MyDenseNet(num_tasks=4, dp1=0.2, dp2=0.03).to(device)
model.load_state_dict(torch.load(f"./MyDense.pth"), strict=False)

# 加载数据集
train_dataset = UltrasoundDataset('../dataset/train/train_feature_data', transform=train_transforms_new, apply_cutmix=True, apply_cutblack=True)
test_dataset = UltrasoundDataset('../dataset/test_A/feature', transform=test_transforms, apply_cutmix=False, apply_cutblack=True)


train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

train(model, train_loader, test_loader, device)

