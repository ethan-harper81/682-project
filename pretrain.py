import torchvision.models as models
import torch.nn as nn
from load_data import load_data

model = models.densenet121(pretrained=False)

num_classes = 18
model.classifier = nn.Linear(model.classifier.in_features, num_classes)

