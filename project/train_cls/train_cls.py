import time
import torch
from torch import optim
from func import train,prepare_data
from ConvNeXtModel import  MyConvNeXt

time_begin = time.time()

#数据集路径
train_root = '../dataset/train/train_cls_data'
test_root = '../dataset/test_A/cls'

#加载dataloader和device
device,train_loader,test_loader = prepare_data(train_root,test_root,use_class_weight=False,batch_size=32,mix =False,cutblack=False)

#加载模型
model_name = "MyConvNeXt"
model = MyConvNeXt(6,0.05,0.6).to(device)
model.load_state_dict(torch.load("./MyConv.pth"))

#选定optimizer和scheduler
optimizer = optim.AdamW(model.parameters(), lr=1e-8,weight_decay=1e-3)
scheduler= optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,10,2,1e-9)

#train函数参数设置
acc_threshold = 0.77 #保存模型参数的正确率阈值
epochs = 300
matrix_per_epoch = False #是否每一轮都可视化混淆矩阵

train(model, train_loader, test_loader, optimizer,scheduler,acc_threshold,model_name,device,matrix_per_epoch, epochs)

time_end = time.time()
print(f"Total time: {(time_end - time_begin)//60}min {(time_end - time_begin)%60} sec")

