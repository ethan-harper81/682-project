import torchvision.models as models
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim
import numpy as np
import time
from load_data import load_data, get_split, load_indices, RelabeledData, get_experiment_data
from utils import convert
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from protonet import get_high_confidence_psuedo_positives

def get_finetune_data(batch_size = 8):
  '''
  return datasets for baseline fine tune and testing
  '''
  nontarget_data, target_data, _, _ = load_data()
  true_neg_dataset = load_indices(nontarget_data, 'splits/true_neg_indices.json')

  few_shot, pos_test = get_split(target_data, split_val=5, map_zero=False, store_indices=False)

  neg_finetune, neg_test = get_split(true_neg_dataset, split_val=5, map_zero=False, store_indices=False)
  neg_finetune = RelabeledData(neg_finetune, {old: 0 for old in range(18)})
  neg_test, unused_neg = get_split(neg_test, split_val=.5,  map_zero=False, store_indices=True)
  neg_test = RelabeledData(neg_test, {old: 0 for old in range(18)})

  finetune_dataset = torch.utils.data.ConcatDataset([few_shot, neg_finetune])
  test_dataset = torch.utils.data.ConcatDataset([pos_test, neg_test])

  # finetune_data = torch.utils.data.DataLoader(finetune_dataset, batch_size = batch_size, shuffle = True)
  # test_data = torch.utils.data.DataLoader(test_dataset, shuffle = True)

  return finetune_dataset, test_dataset

def average_metrics(metrics_list):
  avg_metrics = np.zeros(5)
  for metrics in metrics_list:
    avg_metrics += np.array(metrics)
  avg_metrics /= len(metrics_list)

  print("Accuracy:", avg_metrics[0])
  print("Precision:", avg_metrics[1])
  print("Recall:", avg_metrics[2])
  print("F1 Score:", avg_metrics[3])
  print("ROC AUC:", avg_metrics[4])

def train(train_data, hyperparameters = (1e-4, 10, 8)):
  assert torch.cuda.is_available()
  print(torch.cuda.get_device_name(0))

  device = torch.device("cuda")
  model = models.densenet121(pretrained=False)
  state_dict = torch.load('models/densenet121_pretrain.pth')
  state_dict = {k: v for k, v in state_dict.items() if not k.startswith("classifier")}
  model.load_state_dict(state_dict, strict=False)
  model.classifier = torch.nn.Linear(1024, 2)
  model.to(device)

  '''Model Hyperparemeters'''
  lr, epochs, batch_size = hyperparameters

  #Train
  optimizer = optim.Adam(model.parameters(), lr = lr)
  loss_fn = nn.CrossEntropyLoss()

  start_time = time.perf_counter()
  for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for imgs, labels in train_data:
      imgs, labels = imgs.to(device), labels.to(device)

      optimizer.zero_grad()

      outputs = model(imgs)
      loss = loss_fn(outputs, labels)
      loss.backward()
      optimizer.step()

      total_loss += loss.item()
    
    elapsed_time = time.perf_counter()
    print(f"Epoch {epoch + 1}: loss = {total_loss/len(train_data):.4f}, took: {convert(elapsed_time - start_time)}")
  
  return model

#Test
def test(model, test_data, device = "cuda"):
  model.eval()
  y_true = []
  y_probs = []
  pred_labels = []
  for img, label in test_data:
    img, label = img.to(device), label.to(device)

    with torch.no_grad():
      outputs = model(img)

      _, predicted = torch.max(outputs, 1)
      prob = torch.nn.functional.softmax(outputs, dim = 1)
      pos_prob = prob[:, 1]
      
      y_true.extend(label.cpu().numpy())
      y_probs.extend(pos_prob.cpu().numpy())
      pred_labels.extend(predicted.cpu().numpy())

  return (accuracy_score(y_true, pred_labels), precision_score(y_true, pred_labels), recall_score(y_true, pred_labels),f1_score(y_true, pred_labels), roc_auc_score(y_true, y_probs))

def run_eval(train_data, test_data, n = 5, hyperparameters = None):

  metrics_list = []

  for i in range(n):
    print(f'Begin Training {i}')
    model = train(train_data)
    print(f'Begin Evaluation {i}')
    metrics = test(model, test_data)
    metrics_list.append(metrics)

  print(f'Average Metrics on {n} cycle(s):')
  average_metrics(metrics_list)


if __name__ == '__main__':
  nontarget_data, setB, _,_ = load_data()
  setA = load_indices(nontarget_data, 'splits/true_neg_indices.json')
  test_data, finetune_data, support, query = get_experiment_data(setA, setB)

  finetune_loader = DataLoader(finetune_data, batch_size=8, shuffle = True)
  test_data = DataLoader(test_data, shuffle=False)

  run_eval(finetune_loader, test_data)

  pseudo_positives = get_high_confidence_psuedo_positives(support, query)
  updated_finetune = ConcatDataset([finetune_data, pseudo_positives])

  updated_finetune = DataLoader(updated_finetune, batch_size=8, shuffle= True)

  run_eval(updated_finetune, test_data)

  # finetune_data_set, unused_data = get_finetune_data()

  # finetune_data = DataLoader(finetune_data_set, batch_size=8, shuffle = True)

  # print(len(unused_data))

  # test_data_set, query_data_set = get_split(unused_data, split_val=.5, map_zero=True)

  # test_data = DataLoader(test_data_set, shuffle=False)

  # print(len(test_data))

  # run_eval(finetune_data, test_data)