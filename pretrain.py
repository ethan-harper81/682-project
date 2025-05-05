import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
import time
from load_data import *

def convert(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
     
    return "%d:%02d:%02d" % (hour, minutes, seconds)

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.densenet121(pretrained=False)
num_classes = 18
model.classifier = nn.Linear(model.classifier.in_features, num_classes)
model.to(device)

pretrain_dataset, holdout_dataset, pretrain_map, holdout_map  = load_data()
pretrain_train, pretrain_test = load_train_test(pretrain_dataset, batchsize=100)

# for images, labels in pretrain_train:
#     print(f"Labels: {labels}")
#     print(f"Max: {labels.max()}, Min: {labels.min()}")
#     break

optimizer = optim.Adam(model.parameters(), lr = 1e-4)
loss_fn = nn.CrossEntropyLoss()

#PreTrain
epochs = 10
start_time = time.perf_counter()
for epoch in range(epochs):
  model.train()
  total_loss = 0.0

  for imgs, labels in pretrain_train:
    imgs, labels = imgs.to(device), labels.to(device)

    optimizer.zero_grad()

    outputs = model(imgs)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()

    total_loss += loss.item()
  elapsed_time = time.perf_counter()
  print(f"Epoch {epoch + 1}: loss = {total_loss/len(pretrain_train):.4f}, took: {convert5(elapsed_time - start_time)}")

total = 0
#Test Pretrain
model.eval()
for imgs, labels in pretrain_test:
  imgs, labels = imgs.to(device), labels.to(device)

  with torch.no_grad():
    outputs = model(imgs)

    _, predicted = torch.max(outputs, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')
  