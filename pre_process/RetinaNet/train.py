import argparse

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights

from dataset import CustomImageDataset

def train_loop(model, dataloder, device, optimizer):
    model.train()
    total_sum_loss = 0
    total_cls_loss = 0
    total_breg_loss = 0

    for input_data in dataloder:
        images = input_data['image']
        targets = input_data['target']

        images = [ t.to(device) for t in images]
        targets = [ {'boxes':d['boxes'].to(device), 'labels':d['labels'].to(device)} for d in targets]

        losses = model(images=images, targets=targets)
        cls_loss = losses['classification_loss']
        breg_loss = losses['bbox_regression']
        sum_loss = cls_loss + breg_loss

        optimizer.zero_grad()
        sum_loss.backward()
        optimizer.step()

        total_sum_loss += sum_loss
        total_cls_loss += cls_loss
        total_breg_loss += breg_loss
    
    steps = len(dataloder)
    ave_sum_loss = total_sum_loss / steps
    ave_cls_loss = total_cls_loss / steps
    ave_breg_loss = total_breg_loss / steps

    train_loss_collection = {'sum_loss':ave_sum_loss, 'cls_loss':ave_cls_loss, 'breg_loss':ave_breg_loss}
    
    return train_loss_collection

def eval_loop(model, dataloder, device):
    model.train()
    total_sum_loss = 0
    total_cls_loss = 0
    total_breg_loss = 0

    for input_data in dataloder:
        images = input_data['image']
        targets = input_data['target']

        images = [ t.to(device) for t in images]
        targets = [ {'boxes':d['boxes'].to(device), 'labels':d['labels'].to(device)} for d in targets]

        losses = model(images=images, targets=targets)
        cls_loss = losses['classification_loss']
        breg_loss = losses['bbox_regression']
        sum_loss = cls_loss + breg_loss

        total_sum_loss += sum_loss
        total_cls_loss += cls_loss
        total_breg_loss += breg_loss
    
    steps = len(dataloder)
    ave_sum_loss = total_sum_loss / steps
    ave_cls_loss = total_cls_loss / steps
    ave_breg_loss = total_breg_loss / steps
    
    valid_loss_collection = {'sum_loss':ave_sum_loss, 'cls_loss':ave_cls_loss, 'breg_loss':ave_breg_loss}
    
    return valid_loss_collection

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--pre_trained', type=bool, default=True)
    parser.add_argument('--num_classes', type=int)
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

    pre_trained = args.pre_tained
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
    
    model.train()

    train_dataset = CustomImageDataset()
    valid_dataset = CustomImageDataset()
    
    train_dataloder = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
    valid_dataloder = DataLoader(dataset=valid_dataset, batch_size=valid_batch_size, shuffle=True)

    if optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == 'Adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_history = {
    'train': {'total_loss': [], 'cls_loss': [], 'reg_loss': []},
    'valid': {'total_loss': [], 'cls_loss': [], 'reg_loss': []}
    }

    for epoch in range(max_epochs):
        train_losses = train_loop(model=model, dataloder=train_dataloder, device=device, optimizer=optimizer)
        loss_history['train']['total_loss'].append(train_losses['sum_loss'])
        loss_history['train']['cls_loss'].append(train_losses['cls_loss'])
        loss_history['train']['reg_loss'].append(train_losses['breg_loss'])

        valid_losses = eval_loop(model=model, dataloder=valid_dataloder, device=device)
        loss_history['valid']['total_loss'].append(valid_losses['sum_loss'])
        loss_history['valid']['cls_loss'].append(valid_losses['cls_loss'])
        loss_history['valid']['reg_loss'].append(valid_losses['breg_loss'])
