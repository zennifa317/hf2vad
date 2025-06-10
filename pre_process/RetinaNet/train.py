import argparse

import torch
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--pre_trained', type=bool, default=True)
    parser.add_argument('--num_classes', type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--optimizer', default='SGD', choices=['SGD', 'Adamw'])
    parser.add_argument('--lr','--learning_rate', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    args = parser.parse_args()

    pre_trained = args.pre_tained
    num_classes = args.num_classes
    device = args.device
    epochs = args.epochs
    optimizer_type = args.optimizer
    lr = args.lr
    momentum = args.momentum
    weight_decay = args.weight_decay

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

        print("--- バックボーンのパラメータを凍結 ---")
        for param in model.backbone.parameters():
            param.requires_grad = False
        print("凍結完了！\n")
    
    else:
        model = retinanet_resnet50_fpn_v2(num_classes=num_classes)
    
    model.train()

    if optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == 'Adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
    