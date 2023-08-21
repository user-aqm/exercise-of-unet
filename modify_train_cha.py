import logging
import os.path
import torch
import torch.nn as nn
import re
from torch import cuda
from torch.utils import data
from dataloader.Loader import Loader
from evaluate import evaluate
from file_save_read import save_arrays_to_file
from util.testNet_channel import UNet
from util.testNet_s import init_weights
from loss import SoftDiceLoss, dice_coefficient
import matplotlib.pyplot as plt
device = torch.device("cuda:0")

'''
    数据加载，预处理在此步骤中加入
'''
train_size=30
test_size=30
save_dir = './realData'
loader = data.DataLoader(Loader('train', save_dir), batch_size=train_size, shuffle=True, num_workers=0,
                         pin_memory=True)
loader1 = data.DataLoader(Loader('val', save_dir), batch_size=test_size, shuffle=True, num_workers=0, pin_memory=True)

def change(output, threshold=0.5):
    output[output > threshold] = 1
    output[output <= threshold] = 0
    return output

"""
    自定义的学习率下降
"""
def adjust_lr(optimizer, lr_gamma=0.1):
    for (i, param_group) in enumerate(optimizer.param_groups):
        param_group['lr'] = param_group['lr'] * lr_gamma
    return optimizer.state_dict()['param_groups'][0]['lr']


"""
    加载模型及获取上次训练的loss值
"""
model_path= "model/test_cha_loss_0.025222.pth"
if os.path.exists(model_path):
    model=torch.load(model_path)
    print("model is run")

    match = re.search(r"(\d+\.\d+)", model_path)#找到指定路径中文件名中所有的数字
    if match:
        number_str = match.group(1)
        n_loss = float(number_str)
        print("Extracted loss value:", n_loss)
    else:
        n_loss = 1      #用来对比每次loss值的，保存最小loss的模型参数
        print("No number found in the file name.")
else:
    model= UNet()
    model= init_weights(model)
    print("init model")

    n_loss = 1

    logging.debug("messgae")

# model = UNet()
# model = init_weights(UNet())

"""
    损失函数：
        1，BCEWithLogitsLoss
        2,softdiceloss+交叉熵
"""
loss_fn=nn.BCEWithLogitsLoss()
loss1 = SoftDiceLoss()
loss2 = nn.CrossEntropyLoss()
alpha = 0.5


"""
  优化器：
    1，SGD
    2，自适应
"""
# optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.99, weight_decay=0.0005)  #
optimizer=torch.optim.Adam(model.parameters())

total_train_step=0
total_test_step=0
total_test_loss=0
epoch=60
lr_decay = 5


if cuda:
    model = model.to(device)
    model.train()


name_loss=0


p_step = []
p_loss = []

# for i in range(epoch):
#     print("___第{}轮训练___".format(i+1))
#
#     # if epoch % lr_decay == 0:
#     #     lr = adjust_lr(optimizer)
#
#     for data in loader:
#         imgs, target = data
#
#         if cuda:
#             imgs = imgs.to(device)
#             target = target.to(device)
#
#         output = model(imgs)
#         # l_1 = loss1(output,target)
#         # l_2 = loss2(output,target)
#         # loss = alpha*l_1 + (1-alpha) * l_2
#         loss = loss_fn(output,target)
#
#         optimizer.zero_grad()
#         loss.backward()
#         # nn.utils.clip_grad_norm(model.parameters(),1)    #梯度裁剪
#         optimizer.step()
#
#         total_train_step=total_train_step+1
#         if total_train_step % 10 == 0:
#             print("训练次数：{}，loss={}".format(total_train_step,loss.item()/train_size))
#             # print("训练次数：{}，loss={}".format(total_train_step,loss.item()/train_size))
#         name_loss=loss.item()/train_size
#         if total_train_step %5 ==0:
#             p_step.append(total_train_step)
#             p_loss.append(loss.item())
#
#         if (i+1) % 10 == 0:             #保存损失函数最小的模型
#             if name_loss <= n_loss:     #compare and  save the miniamun model
#                 n_loss=name_loss
#                 fileName = f"model/Dtest_cha_loss_{n_loss:.6f}.pth"
#                 torch.save(model,fileName)
#                 print(f"model data has been saved to '{fileName}'.")
#                 # torch.save(model, f"./model/model_XGs_{n_loss:.5f}.pth")
#
#
#
# fileName = f"TrainData/TD_Dtest_cha_loss_{n_loss:.6f}.txt"
# save_arrays_to_file(fileName,p_loss,p_step)
#
# plt.plot(p_step,p_loss)
# plt.ylim(0.5,1)
# plt.ylabel('loss')
# plt.show()

v_i = []
v_loss = []
v_acc = []
model.eval()
with torch.no_grad():
    i=0
    dice,jc,asd,hd95 = 0,0,0,0
    total_test_accuracy=0
    for data in loader1:
        imgs, target = data

        if cuda:
            imgs = imgs.to(device)
            target = target.to(device)

        output = model(imgs)
        # l_1 = loss1(output,target)
        # l_2 = loss2(output,target)
        # loss = alpha*l_1 + (1-alpha) * l_2
        loss = loss_fn(output,target)


        output = change(output)
        print("The test loss is {}".format(loss / test_size))

        # total_test_accuracy = dice_coefficient(output,target) * 100
        d,j,a,h = evaluate(output,target)
        dice += d
        jc += j
        asd += a
        hd95 += h


        i = i+1
        # v_i.append(i)
        # v_loss.append(loss/test_size)
        # v_acc.append(total_test_accuracy/test_size)
print("dice is {}".format(dice/i))
print("IOU is {}".format(jc/i))
print("asd is {}".format(asd/i))
print("The  hausdorff distance is {}".format(hd95/i))
# plt.plot(v_i,v_loss)
# plt.plot(v_i,v_acc)
# plt.show()