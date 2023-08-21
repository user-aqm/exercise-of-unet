import torch
from  medpy import metric
import numpy

def evaluate(predict,target):
    if torch.is_tensor(predict):
        predict = predict.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    hd95 = metric.binary.hd95(predict,target)
    return hd95