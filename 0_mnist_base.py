#! /usr/bin/python3

import torch
import torch.nn as nn
#import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch_optimizer as optim
#from optim import Adahessian

# 检查CUDA是否可用
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
print("Using device:", device)

# 定义一个简单的DNN模型
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

# 设置超参数
batch_size = 256
learning_rate = 0.01
num_epochs = 50

# 加载MNIST数据集
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换为Tensor
    transforms.Normalize((0.1307,), (0.3081,))  # 归一化
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型、损失函数和优化器
model = SimpleDNN().to(device)
model.train()
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4)
optimizer = optim.Adahessian(model.parameters(), lr=learning_rate, weight_decay=1e-4)
#optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# 训练模型
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 将数据移动到GPU
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        #loss.backward()
        loss.backward(create_graph=True)
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# 测试模型
model.eval()  # 将模型设置为评估模式
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        # 将数据移动到GPU
        images, labels = images.to(device), labels.to(device)

        outputs = model(images, False)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')

# 保存模型
torch.save(model.state_dict(), 'simple_dnn_mnist.pth')
