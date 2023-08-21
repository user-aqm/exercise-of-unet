import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class Mydataset(Dataset):
    def __int__(self,data,targets):
        self.data=data
        self.targets=targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x=self.data[index]
        y=self.targets[index]
        return x,y

# if __name__ == '__main__':
    # dataset=Mydataset('../data/train/image','../data/train/label')
    # dataLoader=DataLoader(dataset,batch_size=10,shuffle=True)
    #
    # print(dataLoader.size)
