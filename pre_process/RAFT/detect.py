import argparse

import numpy as np
import torch
from torchvision.models.optical_flow import raft_large
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms import v2

from dataset import RAFTDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='使用するデバイス')
    parser.add_argument('--frame_dir', required=True, help="連続フレーム画像が含まれるディレクトリ")
    parser.add_argument('--batch_size', type=int, default=4, help="推定時のバッチサイズ")
    args = parser.parse_args()

    device = args.device
    frame_dir = args.frame_dir
    batch_size = args.batch_size

    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print('CUDA is not available, using CPU instead.')
    
    print(f'Using device: {device}')

    model = raft_large(pretrained=True).to(device)
    model.eval()
    
    print("RAFT model loaded successfully.")

    dataset = RAFTDataset(root_dir=frame_dir, transforms=None)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad(): # 推論なので勾配計算は不要！
        for img1_batch, img2_batch, path_batch in tqdm(dataloader):
            
            img1_batch = img1_batch.to(device)
            img2_batch = img2_batch.to(device)
            
            # モデルでフローを推定
            predicted_flow = model(img1_batch, img2_batch)
            
            # バッチ内の結果を1つずつ処理
            # for i in range(len(predicted_flow)):
            #     flow = predicted_flow[i]
            #     original_path = path_batch[i]
                
            #     # ... ここで推定したフローを可視化したり、.floファイルで保存したりする処理 ...
            #     # save_flow_as_image(flow, original_path)
            pass

    print("Estimation complete!")