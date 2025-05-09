import torch
import torchvision.models as models
from torch.utils.data import Subset, ConcatDataset, DataLoader
import torch.nn.functional as F
from load_data import get_split, load_data, load_indices, RelabeledData
import random

class ProtoNet(torch.nn.Module):
  def __init__(self, backbone):
    super().__init__()
    self.backbone = backbone
    self.backbone.classifier = torch.nn.Identity()

  def compute_prototypes(self, support_embeddings, support_labels):
    classes = torch.unique(support_labels)
    prototypes = []
    for c in classes:
      class_embeddings = support_embeddings[support_labels == c]
      prototype = class_embeddings.mean(dim=0)
      prototypes.append(prototype)
    return torch.stack(prototypes), classes

  def forward(self, support_samples, support_labels, query_samples):

    support_embeddings = self.backbone(support_samples)
    query_embeddings = self.backbone(query_samples)

    prototypes, class_ids = self.compute_prototypes(support_embeddings, support_labels)
    dists = torch.cdist(query_embeddings, prototypes)
    log_probs = F.log_softmax(-dists, dim = 1)
    return log_probs, class_ids

def get_fewshot_data(n = 5, seed = 100):
  nontarget_data, target_data, _, _ = load_data()
  true_neg_dataset = load_indices(nontarget_data, 'splits/true_neg_indices.json')

  few_shot, unlabeled = get_split(target_data, split_val=n, map_zero = False, store_indices=False)#unlabeled is query but might not want it all

  random.seed()
  neg_support_inds = random.sample(range(len(true_neg_dataset)), n*2)
  unused_neg_inds = [x for x in range(len(true_neg_dataset)) if x not in neg_support_inds]

  neg_support_data = Subset(true_neg_dataset, neg_support_inds)
  unused_neg_data = Subset(true_neg_dataset, unused_neg_inds)

  neg_support_labels = []
  for sample, label in neg_support_data:
    if label not in neg_support_labels:
      neg_support_labels.append(label)

  neg_support_map = {old:0 for old in neg_support_labels}
  neg_support_data = RelabeledData(neg_support_data, neg_support_map)

  support_data = ConcatDataset([few_shot, neg_support_data])

  query = ConcatDataset([unlabeled, unused_neg_data])

  return support_data, query

def extract_samples(data, device = "cuda"):
  samples, labels = [], []

  for sample, label in data:
    samples.append(sample.to(device))
    labels.append(label.to(device))
  
  return torch.cat(samples), torch.cat(labels)
  
def run_proto(model, support_data, query_data, confidence = .9):
  support_samples, support_labels = extract_samples(support_data)
  query_samples, _ = extract_samples(query_data)

  with torch.no_grad():
    log_probs, class_ids = model(support_samples, support_labels, query_samples)
    probs = torch.exp(log_probs)
    preds = torch.argmax(probs, dim = 1)


    pos_class_idx = (class_ids == 1).nonzero(as_tuple=True)
    pos_probs = probs[:, pos_class_idx]

    high_confidence_inds = (pos_probs >= confidence).squeeze()
    pseudo_positives = query_samples[high_confidence_inds]
    pseudo_probs = pos_probs[high_confidence_inds]
  
  print(f"Selected {len(pseudo_positives)} pseudo-labeled positives with prob ≥ {confidence:.2f}")
  return pseudo_positives, pseudo_probs, high_confidence_inds

def get_high_confidence_psuedo_positives(support, query):
  #support, query, unused_neg = get_fewshot_data()
  support_data = DataLoader(support)
  query_data = DataLoader(query)

  backbone = models.densenet121(pretrained=False)
  state_dict = torch.load('models/densenet121_pretrain.pth')
  state_dict = {k: v for k, v in state_dict.items() if not k.startswith("classifier")}
  backbone.load_state_dict(state_dict, strict=False) 

  model = ProtoNet(backbone)
  model.to("cuda")


  psuedo_positive, psuedo_probs, high_confidence_mask = run_proto(model, support_data, query_data)

  high_confidence_inds = []
  for i, j in enumerate(high_confidence_mask):
    if j:
      high_confidence_inds.append(i)

  high_conf_pos = Subset(query, high_confidence_inds)

  labels = []
  for sample, label in high_conf_pos:
    if label not in labels:
      labels.append(label)

  high_conf_pos = RelabeledData(high_conf_pos, {old:1 for old in labels})

  return high_conf_pos

if __name__ == "__main__":
  support, query = get_fewshot_data()
  print(len(query))
  
  psuedo_positives = get_high_confidence_psuedo_positives(support, query)

  labels = []
  for sample, label in psuedo_positives:
    if label not in labels:
      labels.append(label)
      print(label)

  print(len(psuedo_positives))

  
  
 