import argparse

import torch
from torchvision.models.optical_flow import raft_large

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run the model on')
    args = parser.parse_args()

    device = args.device

    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print('CUDA is not available, using CPU instead.')
    
    print(f'Using device: {device}')

    model = raft_large(pretrained=True).to(device)
    model.eval()

    print("RAFT model loaded successfully.")