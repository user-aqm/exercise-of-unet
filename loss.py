import torch
import torch.nn as nn
import torch.nn.functional as F

#损失函数
class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1

        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score

def dice_coefficient(predict,target,epsilon=1e-6):   #acc crdict 评价指标
    intersection = torch.sum(predict * target)
    union = torch.sum(predict) + torch.sum(target)
    dice = (2 * intersection + epsilon) / ( union + epsilon)
    return dice


