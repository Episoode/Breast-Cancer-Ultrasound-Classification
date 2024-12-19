import torch
from torch.utils.data import DataLoader
import csv
from model.DenseNetModel import MyDenseNet
from model.ConvNeXtModel import MyConvNeXt
from model.data import UltrasoundDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 1 # 设置为单独处理

# 六分类
cla_dir = '../testB/cla/'

cla_dataset = UltrasoundDataset(cla_dir)
cla_data_loader = DataLoader(cla_dataset, batch_size=batch_size, shuffle=False)

# 加载模型
cla_model = MyConvNeXt(6, 0.03, 0.6).to(device)
cla_model.load_state_dict(torch.load("model/MyConv.pth"))
cla_model.eval()

cla_output_csv = 'cla_pre.csv' # 分类预测结果目标文件

with torch.no_grad():
    results = []
    for image_name, images in cla_data_loader:
        images = images.to(device)
        outputs = cla_model(images)
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu().numpy()

        for pred in preds:
            results.append([image_name[0], pred])

    with open(cla_output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['img_name', 'label'])
        writer.writerows(results)

# 特征四任务二分类
fea_dir = '../testB/fea/'

fea_dataset = UltrasoundDataset(fea_dir)
fea_data_loader = DataLoader(fea_dataset, batch_size=batch_size, shuffle=False)

# 加载模型
model_fea = MyDenseNet().to(device)
model_fea.load_state_dict(torch.load("model/MyDense.pth"))

fea_output_csv = 'fea_pre.csv' # 特征分类预测结果目标文件

model_fea.eval()

# 开始预测
with torch.no_grad():
    results = []
    
    for image_name, images in fea_data_loader:
        images = images.to(device)
        
        # 模型预测输出 shape: [batch_size, 4, num_classes]
        outputs = model_fea(images)  
        
        # 获取每个任务的预测类别
        preds = torch.argmax(outputs, dim=2)  # preds shape: [batch_size, 4]
        
        # 将每个样本的预测结果格式化
        for pred in preds.cpu().numpy():
            # 将样本 id 和四个任务的预测结果存入 results 列表
            results.append([image_name[0]] + pred.tolist())

# 写入预测结果到 csv 文件
with open(fea_output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['img_name', 'boundary', 'calcification', 'direction', 'shape'])
    writer.writerows(results)
