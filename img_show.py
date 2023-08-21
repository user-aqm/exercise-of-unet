import matplotlib.pyplot as plt
import torch
from PIL import Image
from util.testNet_s import UNet,init_weights

import torchvision.transforms.functional as f

import matplotlib.pyplot as plt
device = torch.device("cuda")
model_path= "model/Dtest_s_loss_0.024913.pth"
model=torch.load(model_path)



img_path = "realData/val/image/0002.png"
image = Image.open(img_path)

label_path = "realData/val/label/0002.png"
label = Image.open(label_path)
# label = image.convert("RGB")

img=f.to_tensor(image)
img = img.unsqueeze(0)

model.eval()


# model.to(device)
# img=img.to(device)
re_image=model(img)

re_image= re_image.squeeze(0)
re_image=f.to_pil_image(re_image)

plt.subplot(1,3,1)
image = image.convert("RGBA")
plt.imshow(image)
plt.title('orgin image')


# re_image= re_image.point(lambda x:255 if x>=128 else 0,'1')
colors = [(0,0,1,0),(0,0,1,1)]
cmap = plt.matplotlib.colors.ListedColormap(colors)
# cmap = plt.cm.get_cmap("RdYlBu",2)
plt.subplot(1,3,3)
plt.imshow(re_image,cmap=cmap,interpolation="nearest",vmin=0,vmax=1,alpha=1)
plt.imshow(image,cmap="gray",interpolation="nearest",alpha=0.7)
plt.title('result image')

label= label.point(lambda x:255 if x>=128 else 0,'1')
plt.subplot(1,3,2)
plt.imshow(label)
plt.title('orgin label')

plt.show()