import time
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn
from torch.utils.data import WeightedRandomSampler, DataLoader
from data_cls import UltrasoundDataset, train_transforms, test_transforms
from torch.cuda.amp import GradScaler, autocast


def prepare_data(train_root, test_root, use_class_weight=True, batch_size=32, mix=False, cutblack=False):
    '''
    参数说明:
        train_root, test_root, use_class_weight=True, batch_size=32, mix=False, cutblack=False
        train_root, test_root分别为训练集和测试集的路径
        use_class_weight指定是否使用过采样
        batch_size指定批量大小
        mix指定图像增广中是否是否使用cutmix处理
        cutblack指定图像增广中是否是否使用cutblack处理
    功能说明:
        返回device, train_loader, test_loader
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = UltrasoundDataset(train_root, transform=train_transforms, CutMix=mix, cutmix_prob=0.8,CutBlack=cutblack)
    test_dataset = UltrasoundDataset(test_root, transform=test_transforms, CutMix=False, cutmix_prob=0.8,CutBlack=cutblack)

    if use_class_weight: #使用过采样策略
        class_weights = torch.tensor([1, 1, 10, 5, 5, 2])
        sample_weights = [class_weights[label] for label in train_dataset.labels]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:#不适用过采样策略
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return device, train_loader, test_loader


def train(model, train_loader, test_loader, optimizer, scheduler, acc_threshold,
          model_name, device, matrix_per_epoch,epochs=200):
    '''
    参数说明:
        model
        train_loader, test_loader
        optimizer, scheduler  scheduler调整学习速率
        acc_threshold 保存模型参数的正确率最低值
        model_name
        device
        matrix_per_epoch 是否每一轮都显示混淆矩阵
        epochs
    功能说明:
        在混合精度下训练模型,达到准确率要求时保存正确率最高的模型参数，每一轮打印测试集，训练集上的loss和正确率
        最后可视化混淆矩阵以及正确率和loss曲线
    '''

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.2).to(device)
    model.train()
    max_acc = 0
    train_losses, train_accs, test_losses, test_accs = [], [], [], []
    scaler = GradScaler()

    for epoch in range(epochs):
        tot_loss, correct, total, time1 = 0.0, 0, 0, time.time()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = loss_fn(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tot_loss += loss.item()
            predict = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predict == labels).sum().item()

        if scheduler is not None:
            scheduler.step()

        time2 = time.time()
        train_acc = correct / total
        train_loss = tot_loss / len(train_loader)
        test_acc, test_loss,f1, y_true, y_pred = test(model, loss_fn, test_loader, device)
        if max_acc < test_acc:
            max_acc = test_acc
            if max_acc > acc_threshold:
                torch.save(model.state_dict(), f'{model_name}_{100 * max_acc:.2f}.pth')

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        epoch_time = time2 - time1

        print(f"Epoch [{epoch + 1}/{epochs}], "
              f"Train Loss: {train_loss:.4f}, "
              f"Train Accuracy: {100 * train_acc:.2f}%, "
              f"Test Loss: {test_loss:.4f}, "
              f"Test Accuracy: {100 * test_acc:.2f}%, "
              f"Test F1: {100 * f1:.2f}, "
              f"Epoch time: {epoch_time:.2f}s")
        if matrix_per_epoch:
            plot_conf_matrix(confusion_matrix(y_true, y_pred),
                             classes=['2', '3', '4A', '4B', '4C', '5'],
                             title='Confusion Matrix')

    visualize(train_losses, train_accs, test_losses, test_accs)
    plot_conf_matrix(confusion_matrix(y_true, y_pred),
                     classes=['2', '3', '4A', '4B', '4C', '5'],
                     title='Confusion Matrix')


def test(model, loss_fn, test_loader, device):
    '''
    功能说明:
        计算测试集上的准确率，loss，F1分数
    '''
    model.eval()
    correct, total, tot_loss = 0, 0, 0.0
    y_true, y_pred = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            with autocast():  # 使用混合精度
                test_outputs = model(imgs)
                loss = loss_fn(test_outputs, labels)

            predict = torch.argmax(test_outputs, dim=1)
            total += labels.size(0)
            correct += (predict == labels).sum().item()
            tot_loss += loss.item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predict.cpu().numpy())

    # 计算准确率、平均损失和宏平均的 F1 分数
    accuracy = correct / total
    avg_loss = tot_loss / len(test_loader)
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    model.train()
    return accuracy, avg_loss, macro_f1, y_true, y_pred


def visualize(train_losses, train_accs, test_losses, test_accs):
    '''
    功能说明:
        可视化训练集，测试集上的正确率和loss曲线
    '''
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Loss
    ax1.plot(epochs, train_losses, color='tab:red', label='Train Loss')
    ax1.plot(epochs, test_losses, color='tab:blue', label='Test Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Train and Test Loss')
    ax1.legend(loc='upper right')

    # Accuracy
    ax2.plot(epochs, train_accs, color='tab:blue', label='Train Accuracy')
    ax2.plot(epochs, test_accs, color='tab:orange', label='Test Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Train and Testing Accuracy')
    ax2.legend(loc='lower right')

    plt.tight_layout()
    plt.show()


def plot_conf_matrix(cm, classes=None, normalize=True, title='Confusion Matrix', cmap=plt.cm.Blues):
    '''
    功能说明:
        可视化六分类的混淆矩阵
    '''
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes)) if classes is not None else np.arange(cm.shape[0])
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show(block=False)