import os
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
from torch.utils.data.dataset import Dataset
from torchvision.transforms import *


class Loader(Dataset):
    def __init__(self, split, save_dir):
        image_dir = os.path.join(save_dir, split, 'image')
        label_dir = os.path.join(save_dir, split, 'label')
        self.images, self.labels = self._read_data(image_dir, label_dir)
        self.trans = Compose([
            ToPILImage(),
            RandomHorizontalFlip(0.5),#50%的概率进行水平翻转
            RandomVerticalFlip(0.5),
            # RandomResizedCrop(512),#随机裁剪成572
            ToTensor()])

    def _read_data(self, image_dir, label_dir):
        images, labels = [], []
        img_fns = os.listdir(image_dir)
        for img_fn in img_fns:
            image_path = os.path.join(image_dir, img_fn)
            label_path = os.path.join(label_dir, img_fn)
            # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) / 255.
            image_array = Image.open(image_path).convert('L')
            image = np.array(image_array) / 255
            images.append(image[np.newaxis, :])                                #添加一个新的维度，将其转换为三维数组，使得其形状变为 (1, height, width)，
            # label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE) / 255.

            label_array = Image.open(label_path).convert('L')
            label = np.array(label_array) / 255

            label[label > 0.5] = 1
            label[label <= 0.5] = 0
            labels.append(label[np.newaxis, :])
        return images, labels

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        if np.random.uniform(0, 1) < 0.5:
            image = image[:, ::-1, :]
            label = label[:, ::-1, :]
        if np.random.uniform(0, 1) < 0.5:
            image = image[:, :, ::-1]
            label = label[:, :, ::-1]

        # 是将numpy数组转换为PyTorch的浮点型张量，并保证数据的连续性
        image = np.ascontiguousarray(image)#转换为连续存储的数组
        label = np.ascontiguousarray(label)
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()#numpy数组转换为对应的张量。

        return image, label

    def __len__(self):
        return len(self.images)

if __name__=='__main__':
    save_dir = '../realData'
    loader = data.DataLoader(Loader('train', save_dir), batch_size=5, shuffle=True, num_workers=0,
                             pin_memory=True)