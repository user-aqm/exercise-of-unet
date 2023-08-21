import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import re
import torch
import torch.nn as nn
from torch import cuda
from torch.utils import data
from dataloader.Loader import Loader
from file_save_read import save_arrays_to_file
from util.realUNet import UNet
from loss import dice_coefficient
from evaluate import evaluate
import matplotlib.pyplot as plt

from util.testNet_channel import init_weights

device = torch.device("cuda:0")


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
#

# model=UNet()
# model_path="model/model_re.pth"
# if model_path != '':
#     model=torch.load(model_path)
# else:
#     model=UNet()
#     model=model._init_weights()

model_path= "model/model_re_0.022809.pth"
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


loss_fn=nn.BCEWithLogitsLoss()

def adjust_lr(optimizer, lr_gamma=0.1):
    for (i, param_group) in enumerate(optimizer.param_groups):
        param_group['lr'] = param_group['lr'] * lr_gamma
    return optimizer.state_dict()['param_groups'][0]['lr']


# lrate=1e-6
lr_decay = 10

optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.99, weight_decay=0.0005)
# optimizer=torch.optim.Adam()

total_train_step=1
total_test_step=0
total_test_loss=0
epoch=50



if cuda:
    model = model.to(device)
    model.train()
name_loss=0


# p_i = []
# p_loss = []
# for i in range(epoch):
#     print("___第{}轮训练___".format(i+1))
#
#     if epoch % lr_decay == 0:
#         lr = adjust_lr(optimizer)
#
#     model.train()
#     for data in loader:
#         imgs, target = data
#
#         # print(imgs.size)
#
#         if cuda:
#             imgs = imgs.to(device)
#             target = target.to(device)
#
#         output = model(imgs)
#         loss = loss_fn(output,target)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         total_train_step=total_train_step+1
#         if total_train_step % 10 == 0:
#             print("训练次数：{}，loss={}".format(total_train_step,loss.item()/train_size))
#         name_loss=loss.item()/train_size
#         if total_train_step %5 ==0:
#             p_i.append(total_train_step)
#             p_loss.append(loss.item())
#
#         if (i+1) % 10 == 0:             #保存损失函数最小的模型
#             if name_loss <= n_loss:     #compare and  save the miniamun model
#                 n_loss=name_loss
#                 fileName = f"model/model_re_{n_loss:.6f}.pth"
#                 torch.save(model,fileName)
#                 print(f"model data has been saved to '{fileName}'.")
#                 # torch.save(model, f"./model/model_XGs_{n_loss:.5f}.pth")
# #
# fileName = f"TrainData/TD_mod_unet_loss_{n_loss:.6f}.txt"
# save_arrays_to_file(fileName,p_loss,p_i)
#
# plt.ylim(0,1)
# plt.ylabel('loss')
# plt.plot(p_i,p_loss)
# plt.show()


v_i = []
v_loss = []
v_acc = []
model.eval()
with torch.no_grad():
    i=0
    total_test_accuracy=0
    dice,jc,asd,hd95 = 0,0,0,0
    for data in loader1:
        imgs, target = data

        if cuda:
            imgs = imgs.to(device)
            target = target.to(device)

        output = model(imgs)
        loss = loss_fn(output, target)


        # output = (output >= 0.5).astype(int)
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
        # v_acc.append(total_test_accuracy)
print("dice is {}".format(dice/i))
print("IOU is {}".format(jc/i))
print("asd is {}".format(asd/i))
print("The  hausdorff distance is {}".format(hd95/i))
# plt.plot(v_i,v_loss)
# # plt.plot(v_i,v_acc)
# plt.show()