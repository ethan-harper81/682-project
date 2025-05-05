import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
from load_data import load_data

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.densenet121(pretrained=False)

train_loader, test_loader = load_data()

num_classes = 18
model.classifier = nn.Linear(model.classifier.in_features, num_classes)

optimizer = optim.Adam(model.parameters(), lr = 1e-4)

loss_fn = nn.CrossEntropyLoss()

epochs = 10

for epoch in range(epochs):
  model.train()
  total_loss = 0.0

  for imgs, labels in train_loader:
    imgs, labels = imgs.to(device), labels.to(device)

    optimizer.zero_grad()

    outputs = model(imgs)
    loss = loss_fn(outputs, labels)
    loss.backwards()
    optimizer.step()
    total_loss += loss.item()

  print(f"Epoch{epoch + 1}: loss = {total_loss/len(train_loader)}:.4f")