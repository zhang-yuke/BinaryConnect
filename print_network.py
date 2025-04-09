import torch

checkpoint_path = 'runs/WideResNet-28-10/model_best.pth.tar'
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print('Keys in checkpoint:', checkpoint.keys())

if 'state_dict' in checkpoint:
    print('Model parameters:')
    for key, value in checkpoint['state_dict'].items():
        print(f'Layer: {key} | Shape: {value.shape} | Values: {value}')