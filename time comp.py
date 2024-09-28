import torch
import torchvision.models as models
import matplotlib.pyplot as plt

def create_resnet50():
    # 创建一个预训练的ResNet50模型
    model = models.resnet50(pretrained=True)
    return model

def visualize_resnet50_complexity():
    # 创建ResNet50模型
    model = create_resnet50()

    # 获取模型的层数和每层的输出特征图数量
    layers = []
    output_channels = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            layers.append(name)
            output_channels.append(module.out_channels)

    # 计算每层的复杂度
    complexities = [n * n for n in output_channels]

    # 绘制折线统计图
    plt.plot(layers, complexities)
    plt.xlabel('Layer')
    plt.ylabel('Complexity ')
    plt.title('Time Complexity of ')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

# 调用函数绘制折线统计图
visualize_resnet50_complexity()

import torch
import matplotlib.pyplot as plt

class FCNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FCNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, 412)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(412, 312)
        self.relu3 = torch.nn.ReLU()
        self.fc4 = torch.nn.Linear(312, 212)
        self.relu4 = torch.nn.ReLU()
        self.fc5 = torch.nn.Linear(212, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)
        return out

input_size = 1024
hidden_size = 512
num_classes = 3
model = FCNN(input_size, hidden_size, num_classes)

# 计算时间复杂度
complexities = []
for layer in model.children():
    if isinstance(layer, torch.nn.Linear):
        complexity = layer.in_features * layer.out_features
        complexities.append(complexity)
    elif isinstance(layer, torch.nn.ReLU):
        complexity = layer.inplace
        complexities.append(complexity)

# 绘制折线图
plt.plot(range(len(complexities)), complexities, marker='*',color='red')
plt.xlabel('Layer Index')
plt.ylabel('Time Complexity')
plt.title('Time Complexity of FCNN')
plt.show()

import torch
import matplotlib.pyplot as plt

class MDNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers):
        super(MDNN, self).__init__()
        layers = []  # 创建一个空列表来存储层

        # 添加输入层到第一个隐藏层
        layers.append(torch.nn.Linear(input_size, hidden_size))
        layers.append(torch.nn.ReLU())

        # 添加更多的隐藏层
        for _ in range(num_layers - 1):
            layers.append(torch.nn.Linear(hidden_size, hidden_size))
            layers.append(torch.nn.ReLU())

        # 添加最后一个全连接层，输出为num_classes
        layers.append(torch.nn.Linear(hidden_size, num_classes))

        # 使用Sequential将这些层封装起来
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)  # 直接在模型上调用forward函数

# 参数设置
input_size = 1024
hidden_size = 512
num_classes = 3
num_layers = 5

# 创建模型实例
model = MDNN(input_size, hidden_size, num_classes, num_layers)

# 计算时间复杂度
time_complexity = []
for layer in model.model:
    if isinstance(layer, torch.nn.Linear):
        time_complexity.append(input_size * hidden_size)
    elif isinstance(layer, torch.nn.ReLU):
        time_complexity.append(input_size)
    input_size = hidden_size

# 绘制折线图
plt.plot(range(len(time_complexity)), time_complexity,marker='*',color='red')
plt.xlabel('Layer Index')
plt.ylabel('Time Complexity')
plt.title('Time Complexity of MDNN')
plt.show()

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class GSHq_CHN(nn.Module):
    def __init__(self, graph_feat_size):
        super(GSHq_CHN, self).__init__()
        # co-guided networks
        self.w_p_z = nn.Linear(graph_feat_size, graph_feat_size, bias=True)
        self.w_p_r = nn.Linear(graph_feat_size, graph_feat_size, bias=True)
        self.w_p = nn.Linear(graph_feat_size, graph_feat_size, bias=True)

        self.u_i_z = nn.Linear(graph_feat_size, graph_feat_size, bias=True)
        self.u_i_r = nn.Linear(graph_feat_size, graph_feat_size, bias=True)
        self.u_i = nn.Linear(graph_feat_size, graph_feat_size, bias=True)

        self.w_pi_1 = nn.Linear(graph_feat_size, graph_feat_size, bias=True)
        self.w_pi_2 = nn.Linear(graph_feat_size, graph_feat_size, bias=True)
        self.w_c_z = nn.Linear(graph_feat_size, graph_feat_size, bias=True)
        self.u_j_z = nn.Linear(graph_feat_size, graph_feat_size, bias=True)
        self.w_c_r = nn.Linear(graph_feat_size, graph_feat_size, bias=True)
        self.u_j_r = nn.Linear(graph_feat_size, graph_feat_size, bias=True)
        self.w_p = nn.Linear(graph_feat_size, graph_feat_size, bias=True)
        self.u_p = nn.Linear(graph_feat_size, graph_feat_size, bias=True)
        self.w_i = nn.Linear(graph_feat_size, graph_feat_size, bias=True)
        self.u_i = nn.Linear(graph_feat_size, graph_feat_size, bias=True)

    def forward(self, HG1, HG2):
        HG1 = HG1.float()
        HG2 = HG2.float()
        m_c = torch.tanh(self.w_pi_1(HG1 * HG2))
        m_j = torch.tanh(self.w_pi_2(HG1 + HG2))

        r_i = torch.sigmoid(self.w_c_z(m_c) + self.u_j_z(m_j))
        r_p = torch.sigmoid(self.w_c_r(m_c) + self.u_j_r(m_j))

        m_p = torch.tanh(self.w_p(HG1 * r_p) + self.u_p((1 - r_p) * HG2))
        m_i = torch.tanh(self.w_i(HG2 * r_i) + self.u_i((1 - r_i) * HG1))

        HG1_C = (HG1 + m_i) * m_p
        HG2_C = (HG2 + m_p) * m_i

        return HG1_C, HG2_C

graph_feat_size = 1024
model = GSHq_CHN(graph_feat_size)

# Calculate time complexity for each linear layer
time_complexity = [graph_feat_size**2] * len(list(model.named_parameters()))

# Plot the time complexity of each layer
plt.figure(figsize=(10, 5))
plt.bar(range(len(time_complexity)), time_complexity)
plt.xlabel('Layer Index')
plt.ylabel('Time Complexity (O(n^2))')
plt.title('Time Complexity of Each Layer in GSHq_CHN Model')
plt.show()
