import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
import time
from load_data import *
from utils import convert
import os

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.densenet121(pretrained=False)
num_classes = 18
model.classifier = nn.Linear(model.classifier.in_features, num_classes)
model.to(device)

os.makedirs('models/', exist_ok=True)

nontarget_dataset, _,_,_  = load_data()
pretrain_dataset = load_indices(nontarget_dataset, 'splits/pretrain_indices.json')
pretrain_data = load_train_test(pretrain_dataset, train_percent=1, batchsize=128)

optimizer = optim.Adam(model.parameters(), lr = 1e-4)
loss_fn = nn.CrossEntropyLoss()

#PreTrain
epochs = 10
start_time = time.perf_counter()
for epoch in range(epochs):
  model.train()
  total_loss = 0.0

  for imgs, labels in pretrain_data:
    imgs, labels = imgs.to(device), labels.to(device)

    optimizer.zero_grad()

    outputs = model(imgs)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()

    total_loss += loss.item()
  elapsed_time = time.perf_counter()
  print(f"Epoch {epoch + 1}: loss = {total_loss/len(pretrain_data):.4f}, took: {convert(elapsed_time - start_time)}")

torch.save(model.state_dict(), 'models/densenet121_pretrain.pth')

'''
Test Pretrain
Just making sure model works
'''

# total = 0
# correct = 0

# model.eval()
# for imgs, labels in pretrain_test:
#   imgs, labels = imgs.to(device), labels.to(device)

#   with torch.no_grad():
#     outputs = model(imgs)

#     _, predicted = torch.max(outputs, 1)
#     total += labels.size(0)
#     correct += (predicted == labels).sum().item()

# accuracy = 100 * correct / total
# print(f'Test Accuracy: {accuracy:.2f}%')
  