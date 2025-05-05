import numpy as np
import torch
import torchio as tio
import SimpleITK as sitk
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset, random_split
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
  
#Returns 4d image, need to fix
class Resample:
  def __init__(self, resample_transform):
     self.resample_transform = resample_transform


  def __call__(self, img_tensor):
     subject = tio.Subject(image = tio.ScalarImage(tensor=img_tensor.unsqueeze(0)))
     transform = self.resample_transform(subject)

     return transform.image.data
  
class RelabeledData(Dataset):
  def __init__(self, subset, lable_map):
    self.subset = subset
    self.label_map = lable_map

  def __len__(self):
    return len(self.subset)
  
  def __getitem__(self, idx):
     img, label = self.subset[idx]
     return img, self.label_map[label]

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

def load_data(path = "data_set\Medical Imaging Dataset", batch_size = 1, holdout = 'Walker-Warburg'):
  '''
  Loads and transforms data in path.
  Returns: 
    pretrain_dataset: set of all images used for pretraining
    holdout_dataset: set of images used for fewshot and self supervised learning
  '''

  transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    OtsuTransform(), # Background Removal
    transforms.GaussianBlur(kernel_size = 3, sigma = 1.0), #denoising
    #Resample(tio.Resample(1.0)), # Resampling
    ZScoreNormalize(), #intensity Normalization
  ])

  dataset = datasets.ImageFolder(path, transform)

  holdout_indices = [i for i, (path, _) in enumerate(dataset.samples) if holdout in path]
  holdout_label = dataset[holdout_indices[0]][1]
  pretrain_indices = [i for i, (path, _) in enumerate(dataset.samples) if i not in holdout_indices]
  pretrain_labels = [i for i in range(19) if i != holdout_label]

  holdout_map = {holdout_label: 0}
  pretrain_map = {old: new for new, old in enumerate(list(pretrain_labels))}

  pretrain_dataset = Subset(dataset, pretrain_indices)
  holdout_dataset = Subset(dataset, holdout_indices)

  return RelabeledData(pretrain_dataset, pretrain_map), RelabeledData(holdout_dataset, holdout_map), pretrain_map, holdout_map


def load_train_test(dataset, train_percent = .8, batchsize = 1):
  if train_percent == 1:
     return DataLoader(dataset, batch_size=batchsize, shuffle=True)

  train_size =  int(train_percent * len(dataset))
  test_size = len(dataset) - train_size

  train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

  train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
  test_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)

  return train_dataloader, test_dataloader

if __name__ == '__main__':
  p, h, p_map, h_map = load_data()

  p_train, p_test = load_train_test(p,batchsize=32)


  ids = []
  for x,i in p:
    if i not in ids:
       print(i)
       ids.append(i)