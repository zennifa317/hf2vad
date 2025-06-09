import torch
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights

def create_model():
    model = retinanet_resnet50_fpn_v2(weights=None, num_classes=2)

    return model

def load_modelAndWeights(weights_path=None):
    if weights_path is None:
        weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
        model = retinanet_resnet50_fpn_v2(weights=weights)
    else:
        model = create_model()
        model.load_state_dict(torch.load(weights_path))
    
    return model