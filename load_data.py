import numpy as np
import torch
import SimpleITK as sitk
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
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
  
def show_image(img_tensor, title=None):
    # Detach from graph and move to CPU if needed
    img = img_tensor.detach().cpu()

    # Convert to (H, W, C) for Matplotlib
    if img.shape[0] == 3:
        img = img.permute(1, 2, 0)
    elif img.shape[0] == 1:
        img = img.squeeze(0)

    # Display
    plt.imshow(img.numpy(), cmap='gray' if img.ndim == 2 else None)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

def load_data(path = "data_set\Medical Imaging Dataset"):

  transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    OtsuTransform(), # Background Removal
    transforms.GaussianBlur(kernel_size = 3, sigma = 1.0), #denoising
    ZScoreNormalize(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                 std=[0.229, 0.224, 0.225])
  ])

  dataset = datasets.ImageFolder(path, transform)

  dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

  show_image(dataloader.dataset[0][0])

load_data()