import argparse
import glob
import json
import os

import torch
from torchvision.transforms import v2
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from PIL import Image

from RetinaNet import load_modelAndWeights

def detect(img_path, weights_path=None, num_classes=None, score_threshold=0.8):
    if not os.path.exists('detect'):
        os.mkdir('detect') 
    
    with open('./coco_labels.json', mode='r') as f:
        dict_labels = json.load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'using device:{device}')

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True)
        ])

    imgs = []
    exts = ('.jpg', '.png')
    if type(img_path) is str:
        if img_path.endswith(exts):
            img_path = [img_path]
        else:
            img_path = []
            for ext in exts:
                img_path.append(glob.glob(os.path.join(img_path, '*'+ext)))

    for path in img_path:
        img = Image.open(path)
        imgs.append(img)

    if weights_path is None:
        weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
        model = retinanet_resnet50_fpn_v2(weights=weights)
    else:
        model = retinanet_resnet50_fpn_v2(weights=None, num_classes=num_classes)
        model.load_state_dict(torch.load(weights_path))
    
    model.to(device=device)
    model.eval()

    input_imgs = [transform(img).to(device) for img in imgs]
    outputs = model(input_imgs)

    for input_path, img, output in zip(img_path, input_imgs, outputs):
        img_name, ext = os.path.splitext(os.path.basename(input_path))
        de_img_name = 'detected_' + img_name
        output_img = os.path.join('detect', de_img_name+ext)
        output_txt = os.path.join('detect', de_img_name+'.txt')

        boxes=output['boxes'][output['scores'] > score_threshold]
        obj_classes = output['labels'][output['scores'] > score_threshold]
        obj_labels = []
        for obj_class in obj_classes:
            obj_labels.append(dict_labels[str(obj_class.item())])

        img_with_box = draw_bounding_boxes(image=img, boxes=boxes, labels=obj_labels)
        img_with_box = v2.ToPILImage()(img_with_box)
        img_with_box.save(output_img)

        with open(output_txt, mode='w') as g:
            for box, obj_label in zip(boxes, obj_labels):
                box = box.tolist()
                g.write(f'Class: {obj_label}, Box: {box}\n')
    
    print('Detection complete!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--img')
    parser.add_argument('--weights', default=None)
    parser.add_argument('--num_classes', type=int, default=None)
    parser.add_argument('--score_threshold', type=float, default=0.8)

    args = parser.parse_args()

    img_path=args.img
    weights_path=args.weights
    num_classes=args.num_classes
    score_threshold=args.score_threshold

    detect(img_path, weights_path, num_classes, score_threshold)