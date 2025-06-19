import argparse
import json

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from dataset import CustomImageDataset

def test(model, dataloader, device, class_dict):
    metric = MeanAveragePrecision(iou_type="bbox", extended_summary=True)

    with torch.no_grad(): 
        for images, targets in tqdm(dataloader):
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
            
            images = [t.to(device) for t in images]
            preds = model(images)
            metric.update(preds, targets)

    valid_eval = metric.compute()
    
    return valid_eval

def save_results(perf, output_dir):
        import os
        import json
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'eval_results.json'), 'w') as f:
            json.dump(perf, f, indent=4)

def collate_fn(batch): 
    images = [item['image'] for item in batch]
    targets = [item['target'] for item in batch]
    
    return images, targets

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--weights', default=None)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num_classes', type=int)
    parser.add_argument('--test_batch_size', type=int, default=8)
    parser.add_argument('--labels', default='./coco_labels.json')

    args = parser.parse_args()

    data_dir = args.data_dir
    weights_path = args.weights
    output_dir = args.output_dir
    device = args.device
    test_batch_size = args.test_batch_size
    num_classes = args.num_classes
    labels_path = args.labels

    with open(labels_path, mode='r') as f:
        dict_labels = json.load(f)

    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print('CUDAが使えないので、CPUを使用します')
    print(f'Using device : {device}')

    if weights_path is None:
        weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
        model = retinanet_resnet50_fpn_v2(weights=weights)
    else:
        model = retinanet_resnet50_fpn_v2(weights=None, num_classes=num_classes)
        model.load_state_dict(torch.load(weights_path))
    
    model.to(device=device)
    model.eval()

    test_dataset = CustomImageDataset()
    test_dataloder = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=True, collate_fn=collate_fn)

    perf = test(model=model, dataloder=test_dataloder, device=device, class_dict=dict_labels)
    print(perf)    
    
    save_results(perf, output_dir)