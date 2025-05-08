import numpy as np
import torch
import torchio as tio
import SimpleITK as sitk
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import time
from utils import convert
import os
import json


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
  Loads and transforms data in path

  Returns 
  - nontarget_dataset: set of all images belonging to diseases that are not "holdout"
  - target_dataset: set of images for "holdout" disease
  - nontarget_map: maps nontarget disease from original index to index in range(18) - MAY NOT BE USED
  - target_map: maps original target disease index to 0 - MAY NOT BE USED
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

  target_indices = [i for i, (path, _) in enumerate(dataset.samples) if holdout in path]
  target_label = dataset[target_indices[0]][1]
  nontarget_indices = [i for i, (path, _) in enumerate(dataset.samples) if i not in target_indices]
  nontarget_labels = [i for i in range(19) if i != target_label]

  target_map = {target_label: 0}
  nontarget_map = {old: new for new, old in enumerate(list(nontarget_labels))}

  nontarget_dataset = Subset(dataset, nontarget_indices)
  target_dataset = Subset(dataset, target_indices)   

  return RelabeledData(nontarget_dataset, nontarget_map), RelabeledData(target_dataset, target_map), nontarget_map, target_map

def load_indices(dataset, indices_file):
  '''
  Loads dataset containing only provided indices of full dataset
  '''

  with open(indices_file, "r") as f:
    indices = json.load(f)

  subset = Subset(dataset, indices)
   
  return subset

def get_split(dataset, split_val = .1, map_zero = True, store_indices = False, seed = 100):
  '''

  Splits data into two sets, either N of each class or % of each class

  Use split_point = .1 to get 
  - (true_neg, pretrain) as returns with dataset = set A 
  - (true_pos, unlabeled) with dataset = set B

  Use split_point = 5 to get 
  - (neg_finetune, neg_test) as returns with dataset = true_neg 
  - (few_shot, pos_test) with dataset = true_pos
  '''
  split_by_int = isinstance(split_val, int)
  
  zero_map = {old: 0 for old in range(18)}

  class_to_indices = defaultdict(list)
  for idx, (_, label) in enumerate(dataset):
      class_to_indices[label].append(idx)

  # Shuffle and split
  subset1_indices = []
  subset2_indices = []

  random.seed(seed)
  for label, indices in class_to_indices.items():
      random.shuffle(indices)
      
      split_point = split_val if split_by_int else int(len(indices) * split_val) 
      
      subset1_indices.extend(indices[:split_point])
      subset2_indices.extend(indices[split_point:])

  # Create subset objects
  subset1 = Subset(dataset, subset1_indices)
  subset1 = RelabeledData(subset1, zero_map) if map_zero else subset1
  subset2 = Subset(dataset, subset2_indices)

  if store_indices:
     with open("splits/true_neg_indices.json", "w") as f:
        json.dump(subset1_indices, f)
     with open("splits/pretrain_indices.json", "w") as f:
        json.dump(subset2_indices, f) 
  return subset1, subset2

def load_train_test(dataset, train_percent = .8, batchsize = 1):
  '''
  Loads the training and testing set for pretraining

  Parameters
  - dataset: original dataset to be split into train and test
  - train_percent: percent of samples from dataset to be used for training
  - batchsize: batchsize

  Returns
  - trainloader: dataloader for training set, if train_percent set to 1 this is the only return and contains all samples for pretraining
  - testloader: dataloader for test set, only returned if train_percent set to 1 this does not return
  '''
  if train_percent == 1:
     return DataLoader(dataset, batch_size=batchsize, shuffle=True)

  train_size =  int(train_percent * len(dataset))
  test_size = len(dataset) - train_size

  train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

  train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
  test_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)

  return train_dataloader, test_dataloader
'''
Runnning this file will establish the split of pretrain data and true negative data
So there is no need to call get_split outside of this file (its slow)
Running this file assumes that data is already extracted using the get_data.py script
Running this file also assumes that the splits\ directory exists (lazy)
Assumed fie structure:
dataset\Medical Imaging Dataset
 - disease{i} for i in {1, ..., 18}
'''
if __name__ == '__main__':
  start = time.perf_counter()
  p, h, p_map, h_map = load_data()
  time_elapsed = time.perf_counter() - start
  print(f"took: {convert(time_elapsed)} to load data" )
  ten, ninety = get_split(p,  store_indices=True)
  time_elapsed = time.perf_counter() - start
  print(f"took: {convert(time_elapsed)} to load data" )
  print(f"{100 * (len(ninety)/ len(p)):.2f}% pretrain")
  print(f"{100 * (len(ten) / len(p)):.2f}% true negatives")

  #p, h, _, _ = load_data()

  pretrain = load_indices(p, "splits/pretrain_indices.json")
  true_neg = load_indices(p, "splits/true_neg_indices.json")

  print(len(pretrain), len(true_neg))
  


