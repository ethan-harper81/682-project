import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
import time
from load_data import load_data, get_split, load_indices, RelabeledData
from utils import convert
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.densenet121(pretrained=False)
state_dict = torch.load('models/densenet121_pretrain.pth')
state_dict = {k: v for k, v in state_dict.items() if not k.startswith("classifier")}
model.load_state_dict(state_dict, strict=False)
model.classifier = torch.nn.Linear(1024, 2)
model.to(device)

nontarget_data, target_data, _, _ = load_data()
true_neg_dataset = load_indices(nontarget_data, 'splits/true_neg_indices.json')

few_shot, pos_test = get_split(target_data, split_val=5, map_zero=False, store_indices=False)

neg_finetune, neg_test = get_split(true_neg_dataset, split_val=5, map_zero=False, store_indices=False)
neg_finetune = RelabeledData(neg_finetune, {old: 0 for old in range(18)})
neg_test = RelabeledData(neg_test, {old: 0 for old in range(18)})

finetune_dataset = torch.utils.data.ConcatDataset([few_shot, neg_finetune])
test_dataset = torch.utils.data.ConcatDataset([pos_test, neg_test])

'''Model Hyperparemeters'''
lr, epochs, batch_size = 1e-4, 10, 8

finetune_data = torch.utils.data.DataLoader(finetune_dataset, batch_size = batch_size, shuffle = True)
test_data = torch.utils.data.DataLoader(test_dataset, shuffle = True)

#Train
print("--Begin Training--")
optimizer = optim.Adam(model.parameters(), lr = lr)
loss_fn = nn.CrossEntropyLoss()

start_time = time.perf_counter()
for epoch in range(epochs):
  model.train()
  total_loss = 0.0
  for imgs, labels in finetune_data:
    imgs, labels = imgs.to(device), labels.to(device)

    optimizer.zero_grad()

    outputs = model(imgs)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()

    total_loss += loss.item()
  
  elapsed_time = time.perf_counter()
  print(f"Epoch {epoch + 1}: loss = {total_loss/len(finetune_data):.4f}, took: {convert(elapsed_time - start_time)}")

#Test
print("--Begin Evaluation--")
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

print("Accuracy:", accuracy_score(y_true, pred_labels))
print("Precision:", precision_score(y_true, pred_labels))
print("Recall:", recall_score(y_true, pred_labels))
print("F1 Score:", f1_score(y_true, pred_labels))
print("ROC AUC:", roc_auc_score(y_true, y_probs))
print("Confusion Matrix:\n", confusion_matrix(y_true, pred_labels))

# if __name__ == '__main__'():
#   print(len(few_shot))
#   print(len(neg_finetune))

#   inds = []
#   for i, j in few_shot:
#     if j not in inds:
#       print(j)
#       inds.append(j)