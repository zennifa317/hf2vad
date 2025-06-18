import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import CustomImageDataset

from torchvision.datasets import VOCDetection
from torchvision.transforms import v2

class_dict = {'aeroplane':1, 'bicycle':2, 'bird':3, 'boat':4, 'bottle':5,
              'bus':6, 'car':7, 'cat': 8, 'chair':9, 'cow':10, 
              'diningtable':11, 'dog':12, 'horse':13, 'motorbike':14, 'person':15,
              'pottedplant':16, 'sheep':17, 'sofa':18, 'train':19, 'tvmonitor':20}

def cal_loss(model, images, targets):
    losses = model(images=images, targets=targets)
    cls_loss = losses['classification']
    breg_loss = losses['bbox_regression']
    sum_loss = cls_loss + breg_loss

    return sum_loss, cls_loss, breg_loss

def train_loop(model, dataloder, device, optimizer):
    model.train()
    epoch_losses = {'total': 0.0, 'cls': 0.0, 'b_reg': 0.0}

    for images, targets in tqdm(dataloder):
        images = [ t.to(device) for t in images]
        #targets = [{'boxes':d['boxes'].to(device), 'labels':d['labels']} for d in targets ]
        targets_list = []

        for d in targets:
            boxes = []
            labels = []
            for e in d['annotation']['object']:
                box = torch.tensor(list(map(int, (e['bndbox']['xmin'],e['bndbox']['ymin'],e['bndbox']['xmax'],e['bndbox']['ymax']))))
                label = torch.tensor([class_dict[e['name']]])
                boxes.append(box)
                labels.append(label)

            tensor_boxes = torch.stack(boxes, dim=0)
            tensor_labels = torch.cat(labels, dim=0)
            targets_list.append({'boxes':tensor_boxes, 'labels':tensor_labels})
    
        targets = [{'boxes':d['boxes'].to(device), 'labels':d['labels'].to(device)} for d in targets_list]

        sum_loss, cls_loss, breg_loss = cal_loss(model, images, targets)

        optimizer.zero_grad()
        sum_loss.backward()
        optimizer.step()

        epoch_losses['total'] += sum_loss.item()
        epoch_losses['cls'] += cls_loss.item()
        epoch_losses['b_reg'] += breg_loss.item()
    
    steps = len(dataloder)
    for k in epoch_losses.keys():
        epoch_losses[k] /= steps
    
    return epoch_losses

def valid_loop(model, dataloder, device):
    epoch_losses = {'total': 0.0, 'cls': 0.0, 'b_reg': 0.0}
    #metric = MeanAveragePrecision(iou_type="bbox", extended_summary=True)

    with torch.no_grad():
        for images, targets in tqdm(dataloder):
            images = [ t.to(device) for t in images]
            #targets = [{'boxes':d['boxes'].to(device), 'labels':d['labels']} for d in targets ]
            targets_list = []

            for d in targets:
                boxes = []
                labels = []
                for e in d['annotation']['object']:
                    box = torch.tensor(list(map(int, (e['bndbox']['xmin'],e['bndbox']['ymin'],e['bndbox']['xmax'],e['bndbox']['ymax']))))
                    label = torch.tensor([class_dict[e['name']]])
                    boxes.append(box)
                    labels.append(label)

                tensor_boxes = torch.stack(boxes, dim=0)
                tensor_labels = torch.cat(labels, dim=0)
                targets_list.append({'boxes':tensor_boxes, 'labels':tensor_labels})
    
            targets = [{'boxes':d['boxes'].to(device), 'labels':d['labels'].to(device)} for d in targets_list]

            model.train()
            sum_loss, cls_loss, breg_loss = cal_loss(model, images, targets)

            epoch_losses['total'] += sum_loss.item()
            epoch_losses['cls'] += cls_loss.item()
            epoch_losses['b_reg'] += breg_loss.item()
            
            '''model.eval()
            images = [t.to(device) for t in images]
            preds = model(images)
            metric.update(preds, targets)'''
    
    steps = len(dataloder)
    for k in epoch_losses.keys():
        epoch_losses[k] /= steps

    #valid_eval = metric.compute()
    valid_eval = 0
    
    return epoch_losses, valid_eval

def plot_result(output_dir, max_epochs, loss_history):
    total_fig = plt.figure()
    cls_fig = plt.figure()
    breg_fig = plt.figure()
    total_loss = total_fig.add_subplot()
    cls_loss = cls_fig.add_subplot()
    breg_loss = breg_fig.add_subplot()

    x = range(1, max_epochs+1)

    train_total_loss = loss_history['train']['total']
    valid_total_loss = loss_history['valid']['total']
    total_loss.plot(x, train_total_loss, label='Train')
    total_loss.plot(x, valid_total_loss, label='Valid')
    total_loss.set_title('Total Loss')
    total_loss.set_xlabel('Epochs')
    total_loss.set_ylabel('Loss')
    total_loss.legend()
    total_fig.savefig(os.path.join(output_dir, "Total_Losses.png"))

    train_cls_loss = loss_history['train']['cls']
    valid_cls_loss = loss_history['valid']['cls']
    cls_loss.plot(x, train_cls_loss, label='Train')
    cls_loss.plot(x, valid_cls_loss, label='Valid')
    cls_loss.set_title('Classification Loss')
    cls_loss.set_xlabel('Epochs')
    cls_loss.set_ylabel('Loss')
    cls_loss.legend()
    cls_fig.savefig(os.path.join(output_dir, "Class_Losses.png"))

    train_breg_loss = loss_history['train']['b_reg']
    valid_breg_loss = loss_history['valid']['b_reg']
    breg_loss.plot(x, train_breg_loss, label='Train')
    breg_loss.plot(x, valid_breg_loss, label='Valid')
    breg_loss.set_title('Regression Loss')
    breg_loss.set_xlabel('Epochs')
    breg_loss.set_ylabel('Loss')
    breg_loss.legend()
    breg_fig.savefig(os.path.join(output_dir, "Regression_Losses.png"))

def collate_fn(batch):
    return tuple(zip(*batch))

# def collate_fn(batch):
    
#     images = [item['image'] for item in batch]
#     targets = [item['target'] for item in batch]
    
#     return images, targets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--num_classes', type=int)
    parser.add_argument('--pre_trained', type=bool, default=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--valid_batch_size', type=int, default=8)
    parser.add_argument('--optimizer', default='SGD', choices=['SGD', 'Adamw'])
    parser.add_argument('--lr','--learning_rate', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--freeze_backbone', action='store_true')

    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    pre_trained = args.pre_trained
    num_classes = args.num_classes
    device = args.device
    max_epochs = args.epochs
    train_batch_size = args.train_batch_size
    valid_batch_size = args.valid_batch_size
    optimizer_type = args.optimizer
    lr = args.lr
    momentum = args.momentum
    weight_decay = args.weight_decay
    fb = args.freeze_backbone

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print('CUDAが使えないので、CPUを使用します')
    print(f'Using device : {device}')

    if pre_trained:
        weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
        model = retinanet_resnet50_fpn_v2(weights=weights)

        in_channels = model.head.classification_head.cls_logits.in_channels
        num_anchors = model.head.classification_head.num_anchors

        new_cls_logits = torch.nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)

        torch.nn.init.normal_(new_cls_logits.weight, std=0.01)
        torch.nn.init.constant_(new_cls_logits.bias, -4.595) # -log((1 - 0.01) / 0.01)

        model.head.classification_head.cls_logits = new_cls_logits
        model.head.classification_head.num_classes = num_classes

        if fb:
            print("--- バックボーンのパラメータを凍結 ---")
            for param in model.backbone.parameters():
                param.requires_grad = False
            print("凍結完了！\n")
    
    else:
        model = retinanet_resnet50_fpn_v2(num_classes=num_classes)
    
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]

    #train_dataset = CustomImageDataset()
    #valid_dataset = CustomImageDataset()

    train_dataset = VOCDetection(root='./data', year='2012', image_set='train',download=True, transforms = v2.Compose([v2.ToImage(), v2.ToDtype(dtype=torch.float32, scale=True), v2.Resize((608, 1024))]))
    valid_dataset = VOCDetection(root='./data', year='2012', image_set='val',download=True, transforms = v2.Compose([v2.ToImage(), v2.ToDtype(dtype=torch.float32, scale=True), v2.Resize((608, 1024))]))

    train_dataloder = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn)
    valid_dataloder = DataLoader(dataset=valid_dataset, batch_size=valid_batch_size, shuffle=False, collate_fn=collate_fn)

    if optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == 'Adamw':
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    loss_history = {
    'train': {'total': [], 'cls': [], 'b_reg': []},
    'valid': {'total': [], 'cls': [], 'b_reg': []}
    }

    best_valid_loss = float('inf')

    for epoch in range(max_epochs):
        train_losses = train_loop(model=model, dataloder=train_dataloder, device=device, optimizer=optimizer)
        train_total_loss = train_losses['total'].to('cpu').detach().numpy().copy()
        train_cls_loss = train_losses['cls'].to('cpu').detach().numpy().copy()
        train_reg_loss = train_losses['b_reg'].to('cpu').detach().numpy().copy()
        loss_history['train']['total'].append(train_total_loss)
        loss_history['train']['cls'].append(train_cls_loss)
        loss_history['train']['b_reg'].append(train_reg_loss)

        valid_losses, valid_eval = valid_loop(model=model, dataloder=valid_dataloder, device=device)
        valid_total_loss = valid_losses['total'].to('cpu').detach().numpy().copy()
        valid_cls_loss = valid_losses['cls'].to('cpu').detach().numpy().copy()
        valid_reg_loss = valid_losses['b_reg'].to('cpu').detach().numpy().copy()
        loss_history['valid']['total'].append(valid_total_loss)
        loss_history['valid']['cls'].append(valid_cls_loss)
        loss_history['valid']['b_reg'].append(valid_reg_loss)

        print(f'Epoch {epoch+1}/{max_epochs}----------------------------------\n'
              f"Train Loss | Total Loss: {train_total_loss:.4f} Cls Loss: {train_cls_loss:.4f} Breg Loss: {train_reg_loss:.4f}\n"
              f"Valid Loss | Total Loss: {valid_total_loss:.4f} Cls Loss: {valid_cls_loss:.4f} Breg Loss: {valid_reg_loss:.4f}\n")
        #print(valid_eval)

        if valid_total_loss < best_valid_loss:
            best_valid_loss = valid_total_loss
            model_path = os.path.join(output_dir, 'best_model.pth')
            torch.save(model.state_dict(), model_path)
    
    model_path = os.path.join(output_dir, 'last_model.pth')
    torch.save(model.state_dict(), model_path)

    plot_result(output_dir, max_epochs, loss_history)