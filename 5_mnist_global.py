#! /bin/python3

import math
import torch
import torch.nn as nn
import torch.optim as optim
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
print("Using device:", device)

# 加载保存的模型状态
model = SimpleDNN().to(device)
model.load_state_dict(torch.load('simple_dnn_mnist.pth'))

# 定义剪枝函数
def prune(model, prune_ratio, index_dict, sum_es):
    for name, param in model.named_parameters():
        if not name.endswith(".weight"):
            continue
        weight = param
        es = sum_es[index_dict[name]]
        threshold = torch.quantile(es, prune_ratio)
        print("module name={}, threshold={}", name, threshold)
        mask = es < threshold
        with torch.no_grad():
            weight.mul_(mask)

def prune_rows(model, prune_ratio, index_dict, sum_es):
    for name, param in model.named_parameters():
        if not name.endswith(".weight"):
            continue
        weight = param
        es = sum_es[index_dict[name]]
        # 计算每行需要保留的权重数量
        num_weights_to_keep = max(1, int((1 - prune_ratio) * es.size(1)))
        for i in range(es.size(0)):
            row = es[i]
            threshold = torch.topk(row, num_weights_to_keep, largest=True).values.min()
            count = 0
            for j in range(es.size(1)):
                if count >= es.size(1) * prune_ratio:
                    break
                if es[i][j] < threshold:
                    with torch.no_grad():
                        weight[i][j] = 0.0
                    count += 1
            for j in range(es.size(1)):
                if count >= es.size(1) * prune_ratio:
                    break
                if weight[i][j] != 0.0 and es[i][j] == threshold:
                    with torch.no_grad():
                        weight[i][j] = 0.0
                    count += 1
            print(f"module name={name}, i={i}, threshold={threshold}, pruning_ratio={count/float(es.size(1))}")


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=1024, shuffle=True)
criterion = nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4)

model.train()
criterion = nn.CrossEntropyLoss()
l = list(model.parameters())
names = [name for name, param in model.named_parameters()]
index_dict = {name:index for index,name in enumerate(names)}
d = dict(model.named_parameters())

load_from_disk = False
if load_from_disk:
    sum_es = torch.load('sum_es.pt')
else:
    # 累计的E值即累积的二阶导数*参数值的平方/2
    sum_es = []
    for i in range(0, len(names)):
        sum_es.append(torch.zeros_like(l[i]))

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        first_grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        assert len(first_grads) == len(l)
        for i in range(0, len(first_grads)):
            assert (first_grads[i].dim() == 2 or first_grads[i].dim() == 1)
            if first_grads[i].dim() == 2:
                for j in range(0, first_grads[i].size(0)):
                    print(i, j)
                    for k in range(0, first_grads[i].size(1)):
                        second_grad = torch.autograd.grad(first_grads[i][j][k], l[i], create_graph=True)[0][j][k]
                        #print(f"{i},{j},{k},{second_grad.float()}")
                        with torch.no_grad():
                            #sum_es[i][j][k] += math.copysign(1, second_grad.float()) * math.sqrt(abs(second_grad.float())) * abs(l[i][j][k].float())
                            sum_es[i][j][k] += second_grad.float() * l[i][j][k].float() * l[i][j][k].float() / 2.0
            elif first_grads[i].dim() == 1:
                for j in range(0, first_grads[i].size(0)):
                    second_grad = torch.autograd.grad(first_grads[i][j], l[i], create_graph=True)[0][j]
                    #print(f"{i},{j},{second_grad.float()}")
                    with torch.no_grad():
                        #sum_es[i][j] += math.copysign(1, second_grad.float()) * math.sqrt(abs(second_grad.float())) * abs(l[i][j].float())
                        sum_es[i][j] += second_grad.float() * l[i][j].float() * l[i][j].float() / 2.0
        break
    torch.save(sum_es, 'sum_es.pt')


# 剪枝比率
prune_ratio = 0.9  # 假设我们要剪枝50%的权重
#prune_rows(model, prune_ratio, index_dict, sum_es)
prune(model, prune_ratio, index_dict, sum_es)

# 确保模型在评估模式
model.eval()

# 加载MNIST测试数据集
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
