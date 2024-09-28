import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CNN, self).__init__()
        # 定义第一个卷积层，输入通道数为input_size，输出通道数为hidden_size，卷积核大小为3x3，填充为1
        self.conv1 = nn.Conv2d(input_size, hidden_size, kernel_size=3, padding=1)
        # 定义第一个ReLU激活函数
        self.relu1 = nn.ReLU()
        # 定义第二个卷积层，输入通道数为hidden_size，输出通道数为hidden_size，卷积核大小为3x3，填充为1
        self.conv2 = nn.Conv2d(512, 412, kernel_size=3, padding=1)
        # 定义第二个ReLU激活函数
        self.relu2 = nn.ReLU()
        # 定义第三个卷积层，输入通道数为hidden_size，输出通道数为hidden_size，卷积核大小为3x3，填充为1
        self.conv3 = nn.Conv2d(412, 312, kernel_size=3, padding=1)
        # 定义第三个ReLU激活函数
        self.relu3 = nn.ReLU()
        # 定义第四个卷积层，输入通道数为hidden_size，输出通道数为hidden_size，卷积核大小为3x3，填充为1
        self.conv4 = nn.Conv2d(312, 211, kernel_size=3, padding=1)
        # 定义第四个ReLU激活函数
        self.relu4 = nn.ReLU()
        # 定义全连接层，输入特征数为hidden_size * (input_size // 16) * (input_size // 16)，输出类别数为num_classes
        self.fc = nn.Linear(211, num_classes)

    def forward(self, x):
        # 假设输入数据的形状是 [batch_size, channels, height, width]
        # 如果输入数据的形状不是这个形状，请根据实际情况进行调整
        x = x.unsqueeze(1)  # 在通道维度上增加一个维度，使其变为 [batch_size, 1, height, width]
        out = self.conv1(x)  # 通过第一个卷积层
        out = self.relu1(out)  # 应用第一个ReLU激活函数
        out = self.conv2(out)  # 通过第二个卷积层
        out = self.relu2(out)  # 应用第二个ReLU激活函数
        out = self.conv3(out)  # 通过第三个卷积层

        out = self.relu3(out)  # 应用第三个ReLU激活函数
        out = self.conv4(out)  # 通过第四个卷积层
        out = self.relu4(out)  # 应用第四个ReLU激活函数
        out = out.view(out.size(0), -1)  # 将输出展平为一维向量
        out = self.fc(out)  # 通过全连接层得到最终输出

        return out


