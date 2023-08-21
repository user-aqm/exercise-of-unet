import torch
import torch.nn as nn
from torch.utils import data
# from torchsummary import summary
from util import *
from dataloader.Loader import Loader
from util.realUNet import UNet

save_dir = './realData'
device = torch.device("cuda:2")

def accuracy(logit, target, threshold=0.5):
    logit[logit > threshold] = 1
    logit[logit <= threshold] = 0
    return (logit.long() == target.long()).float().mean().item()

def adjust_lr(optimizer, lr_gamma=0.1):
    for (i, param_group) in enumerate(optimizer.param_groups):
        param_group['lr'] = param_group['lr'] * lr_gamma
    return optimizer.state_dict()['param_groups'][0]['lr']

def step(split, epoch, model, criterion, optimizer, batch_size, cuda=False):
    if split == 'train':
        model.train()
    else:
        model.eval()

    loader = data.DataLoader(Loader(split, save_dir), batch_size=batch_size, shuffle=True, num_workers=0,
                             pin_memory=True)
    epoch_loss, epoch_acc, n_batchs = 0, 0, 0
    for i, (image, label) in enumerate(loader):
        n_batchs += image.size(0)
        if cuda:
            image = image.cuda(device)
            label = label.cuda(device)
        logit = model(image)
        logit = logit.flatten()
        label = label.flatten()
        loss = criterion(logit, label)
        if split == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += accuracy(logit, label) * 100
    epoch_loss /= n_batchs
    epoch_acc /= n_batchs
    return epoch_loss, epoch_acc


# %%
batch_size = 32
pretrained = False
cuda = True
start_epoch = 1
end_epoch = 10
lr_decay = 30
# %%
criterion = nn.BCEWithLogitsLoss()
model = UNet()

print(model)


optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.99, weight_decay=0.0005)



# summary(model, (1, 512, 512))


if cuda:
    model = model.cuda(device)
    criterion = criterion.cuda(device)

if pretrained:
    model.load_state_dict(torch.load('model.pth'))
train_losses, val_losses = [], []
for epoch in range(start_epoch, end_epoch):
    if epoch % lr_decay == 0:
        lr = adjust_lr(optimizer)
        print('adjust LR to {:.4f}'.format(lr))
    tepoch_loss, tepoch_acc = step('train', epoch, model, criterion, optimizer, batch_size, cuda=cuda)
    vepoch_loss, vepoch_acc = step('val', epoch, model, criterion, optimizer, batch_size, cuda=cuda)
    train_losses.append(tepoch_loss)
    val_losses.append(vepoch_loss)
    print(
        'epoch {0:} finished, tloss:{1:.4f} [{2:.2f}%]  vloss:{3:.4f} [{4:.2f}%]'.format(epoch, tepoch_loss, tepoch_acc,
                                                                                         vepoch_loss, vepoch_acc))
torch.save(model.state_dict(), 'model.pth')
print('done!')