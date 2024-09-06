#! /bin/python3

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# 定义与之前相同的模型类
class SimpleDNN(nn.Module):
    def __init__(self):
        super(SimpleDNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)  # 28*28是输入图片的像素数，512是隐藏层的神经元数
        self.fc2 = nn.Linear(512, 10)    # 输出层，10个类别
        self.dropout1 = nn.Dropout(0.2)
    def forward(self, x, dropout=True):
        x = x.view(-1, 28*28)  # 将图片展平成一维向量
        x = torch.relu(self.fc1(x))
        if dropout:
            x = self.dropout1(x)  # 在激活函数后应用Dropout
        x = self.fc2(x)
        return x

# 检查CUDA是否可用
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
print("Using device:", device)

# 加载保存的模型状态
model = SimpleDNN().to(device)
model.load_state_dict(torch.load('simple_dnn_mnist.pth'))

# 定义剪枝函数
def prune_model(model, prune_ratio):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data
            threshold = torch.quantile(weight.abs(), prune_ratio)
            print("module name={}, threshold={}", name, threshold)
            mask = weight.abs() < threshold
            module.weight.data.mul_(mask)
            #module.bias.data.mul_(mask.sum(1) > 0)  # 避免偏置全部为0的情况

def prune_columns(model, prune_ratio):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data
            # 计算每列需要保留的权重数量
            num_weights_to_keep = max(1, int((1 - prune_ratio) * weight.size(0)))
            for j in range(weight.size(1)):
                # 对每列进行操作
                column = weight[:, j]
                threshold = torch.topk(column.abs(), num_weights_to_keep, largest=True).values.min()
                count = 0
                for i in range(weight.size(0)):
                    if count >= weight.size(0) * prune_ratio:
                        break
                    if abs(weight[i][j]) < threshold:
                        weight[i][j] = 0
                        count += 1
                for i in range(weight.size(0)):
                    if count >= weight.size(0) * prune_ratio:
                        break
                    if weight[i][j] != 0.0 and abs(weight[i][j]) == threshold:
                        weight[i][j] = 0
                        count += 1
                print(f"module name={name}, i={j}, threshold={threshold}, pruning={count}")

def prune_rows(model, prune_ratio):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data
            # 计算每行需要保留的权重数量
            num_weights_to_keep = max(1, int((1 - prune_ratio) * weight.size(1)))
            for i in range(weight.size(0)):
                # 对每行进行操作
                row = weight[i]
                threshold = torch.topk(row.abs(), num_weights_to_keep, largest=True).values.min()
                count = 0
                for j in range(weight.size(1)):
                    if count >= weight.size(1) * prune_ratio:
                        break
                    if abs(weight[i][j]) < threshold:
                        weight[i][j] = 0.0
                        count += 1
                for j in range(weight.size(1)):
                    if count >= weight.size(1) * prune_ratio:
                        break
                    if weight[i][j] != 0.0 and abs(weight[i][j]) == threshold:
                        weight[i][j] = 0.0
                        count += 1
                print(f"module name={name}, i={i}, threshold={threshold}, pruning_ratio={count/float(weight.size(1))}")

# 剪枝比率
prune_ratio = 0  # 假设我们要剪枝50%的权重
prune_model(model, prune_ratio)
#prune_rows(model, prune_ratio)
#prune_columns(model, prune_ratio)

# 确保模型在评估模式
model.eval()

# 加载MNIST测试数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 测试剪枝后的模型
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images, False)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the pruned network on the 10000 test images: {100 * correct / total} %')
