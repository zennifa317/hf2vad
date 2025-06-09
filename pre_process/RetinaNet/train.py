import torch
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights

def train(pre_trained, num_classes):
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

    
if __name__ == "__main__":
    train()