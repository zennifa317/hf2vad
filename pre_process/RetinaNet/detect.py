import argparse
import glob
import json
import os

import torch
from torchvision.transforms import v2
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm
from PIL import Image

def detect(img_path, labels_path, weights_path=None, num_classes=None, score_threshold=0.8, batch_size=8):
    if not os.path.exists('detect'):
        os.mkdir('detect') 
    
    with open(labels_path, mode='r') as f:
        dict_labels = json.load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'using device:{device}')

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True)
        ])

    image_paths = []
    if os.path.isdir(img_path):
        for ext in ('.jpg', '.png', '.jpeg'):
            image_paths.extend(glob.glob(os.path.join(img_path, f'*{ext}')))
    elif os.path.isfile(img_path):
        image_paths.append(img_path)

    if weights_path is None:
        weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
        model = retinanet_resnet50_fpn_v2(weights=weights)
    else:
        model = retinanet_resnet50_fpn_v2(weights=None, num_classes=num_classes)
        model.load_state_dict(torch.load(weights_path))
    
    model.to(device=device)
    model.eval()

    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i:i + batch_size]

            batch_imgs = [Image.open(p).convert("RGB") for p in batch_paths]
            input_tensors = [transform(img).to(device) for img in batch_imgs]

            outputs = model(input_tensors)

            for input_path, tensor_img, output in zip(img_path, input_tensors, outputs):
                img_name, ext = os.path.splitext(os.path.basename(input_path))
                de_img_name = 'detected_' + img_name
                output_img = os.path.join('detect', de_img_name+ext)
                output_txt = os.path.join('detect', de_img_name+'.txt')

                high_scores_mask = output['scores'] > score_threshold
                boxes = output['boxes'][high_scores_mask]
                labels = output['labels'][high_scores_mask]
                
                str_labels = [dict_labels[str(label.item())] for label in labels]

                img_to_draw = v2.ToDtype(torch.uint8, scale=True)(tensor_img)
                img_with_box = draw_bounding_boxes(image=img_to_draw, boxes=boxes, labels=str_labels)

                pil_img_with_box = v2.ToPILImage()(img_with_box)
                pil_img_with_box.save(output_img)

                with open(output_txt, mode='w') as g:
                    for box, obj_label in zip(boxes, str_labels):
                        box = box.tolist()
                        g.write(f'Class: {obj_label}, Box: {box}\n')
    
    print('Detection complete!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--img')
    parser.add_argument('--weights', default=None)
    parser.add_argument('--num_classes', type=int, default=None)
    parser.add_argument('--score_threshold', type=float, default=0.8)
    parser.add_argument('--labels', default='./coco_labels.json')
    parser.add_argument('--batch_size', type=int, default=8) 

    args = parser.parse_args()

    img_path = args.img
    weights_path = args.weights
    num_classes = args.num_classes
    score_threshold = args.score_threshold
    labels_path = args.labels
    batch_size = args.batch_size

    detect(img_path=img_path, weights_path=weights_path, num_classes=num_classes, score_threshold=score_threshold, labels_path=labels_path, batch_size=batch_size)