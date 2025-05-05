import numpy as np
import torch
import torchio
import SimpleITK as sitk
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt


#Backgroud Rmoval
class OtsuTransform:
  def __call__(self, image):
    out = []
    for c in range(3):
        channel = image[c]
        thresh = threshold_otsu(channel.numpy())
        binary = (channel > thresh).float()
        out.append(binary)
    return torch.stack(out)  
  
class ZScoreNormalize:
    def __call__(self, image) -> torch.Tensor:
        normalized = []
        for c in range(image.shape[0]):
            channel = image[c]
            mean = channel.mean()
            std = channel.std()
            normed = (channel - mean) / (std + 1e-8)  
            normalized.append(normed)
        return torch.stack(normalized)
  
def show_image(img_tensor):
    # Detach from graph and move to CPU if needed
    img = img_tensor.detach().cpu()

    if img.shape[0] == 3:
        img = img.permute(1, 2, 0)
    elif img.shape[0] == 1:
        img = img.squeeze(0)

    plt.imshow(img.numpy(), cmap='gray' if img.ndim == 2 else None)
    plt.axis('off')
    plt.show()

def load_data(path = "data_set\Medical Imaging Dataset", batch_size = 1):

  transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    OtsuTransform(), # Background Removal
    transforms.GaussianBlur(kernel_size = 3, sigma = 1.0), #denoising
    #Resampling,
    #Registrantion,
    ZScoreNormalize(),
  ])

  dataset = datasets.ImageFolder(path, transform)

  print(dataset[1500])

  # pretrain_dataset = Subset(dataset, pretrain_indices)
  # walker_dataset = Subset(dataset, walker_w)

  # pretrain_dataloader = DataLoader(pretrain_dataset, batch_size, shuffle=True)
  # walker_dataloader = DataLoader(walker_dataset, batch_size, shuffle=True)

  dataloader = DataLoader(dataset, batch_size, shuffle = True)

  return dataset, dataloader#pretrain_dataloader, walker_dataloader


dataset, dataloader = load_data()


