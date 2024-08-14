import torch
from transformers import AutoModel

# モデルのロード
model = AutoModel.from_pretrained('output_dir')

# モデルの構造とパラメータの表示
print(model)

for name, param in model.named_parameters():
    print(f'Parameter name: {name}, Shape: {param.shape}')

# パラメータの統計情報を表示
for name, param in model.named_parameters():
    print(f'Parameter name: {name}')
    print(f'Mean: {param.mean().item()}')
    print(f'Stdev: {param.std().item()}')
    print(f'Min: {param.min().item()}')
    print(f'Max: {param.max().item()}')
    print('---')

