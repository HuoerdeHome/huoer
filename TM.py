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

        # enriching the semantics of price and interest preferences
        HG1_C = (HG1 + m_i) * m_p
        HG2_C = (HG2 + m_p) * m_i

        return HG1_C, HG2_C
class FCNN(torch.nn.Module):  # 定义一个名为FCNN的类，继承自torch.nn.Module
    def __init__(self, input_size, hidden_size, num_classes):  # 初始化函数，接收输入层大小、隐藏层大小和输出层类别数作为参数
        super(FCNN, self).__init__()  # 调用父类的初始化函数
        self.fc1 = torch.nn.Linear(input_size, hidden_size)  # 定义第一个全连接层，输入大小为1012，输出大小为512
        self.relu1 = torch.nn.ReLU()  # 定义第一个ReLU激活函数
        self.fc2 = torch.nn.Linear(hidden_size, 412)  # 定义第二个全连接层，输入大小为512，输出大小为512
        self.relu2 = torch.nn.ReLU()  # 定义第二个ReLU激活函数
        self.fc3 = torch.nn.Linear(412, 312)  # 定义第三个全连接层，输入大小为hidden_size，输出大小为hidden_size
        self.relu3 = torch.nn.ReLU()  # 定义第三个ReLU激活函数
        self.fc4 = torch.nn.Linear(312, 212)  # 定义第四个全连接层，输入大小为hidden_size，输出大小为hidden_size
        self.relu4 = torch.nn.ReLU()  # 定义第四个ReLU激活函数
        self.fc5 = torch.nn.Linear(212, num_classes)  # 定义第五个全连接层，输入大小为hidden_size，输出大小为num_classes

    def forward(self, x):  # 定义前向传播函数，接收输入x
        out = self.fc1(x)  # 将输入x传入第一个全连接层
        out = self.relu1(out)  # 对第一个全连接层的输出应用ReLU激活函数
        out = self.fc2(out)  # 将激活后的结果传入第二个全连接层
        out = self.relu2(out)  # 对第二个全连接层的输出应用ReLU激活函数
        out = self.fc3(out)  # 将激活后的结果传入第三个全连接层
        out = self.relu3(out)  # 对第三个全连接层的输出应用ReLU激活函数
        out = self.fc4(out)  # 将激活后的结果传入第四个全连接层
        out = self.relu4(out)  # 对第四个全连接层的输出应用ReLU激活函数
        out = self.fc5(out)  # 将激活后的结果传入第五个全连接层
        return out  # 返回第五个全连接层的输出结果
class CombinedModel(nn.Module):
    def __init__(self, graph_feat_size, input_size, hidden_size, num_classes):
        super(CombinedModel, self).__init__()
        self.gshq_chn = GSHq_CHN(graph_feat_size)
        self.fcnn = FCNN(input_size, hidden_size, num_classes)

    def forward(self, HG1, HG2, x):
        HG1_C, HG2_C = self.gshq_chn(HG1, HG2)
        out = self.fcnn(x)
        return HG1_C, HG2_C, out
def visualize_combined_model_complexity(graph_feat_size, input_size, hidden_size, num_classes, num_layers):
    # 创建CombinedModel实例
    model = CombinedModel(graph_feat_size, input_size, hidden_size, num_classes)

    # 获取模型的参数和操作次数
    parameters, operations = [], []

    # for name, layer in model.named_modules():
    #     if isinstance(layer, (nn.Linear, nn.ReLU)):
    #         parameters.append(layer.weight.numel())
    #         if isinstance(layer, nn.Linear):
    #             operations.append(layer.weight.numel() * layer.bias.numel())
    for layer in model.modules():
        if isinstance(layer, torch.nn.Linear):
            parameters.append(layer.weight.numel())
        elif isinstance(layer, torch.nn.ReLU):
            # 跳过ReLU层，因为它没有权重属性
            continue


    # 计算总参数和操作次数
    total_parameters = sum(parameters)
    total_operations = sum(operations)

    # 输出总参数和操作次数
    print(f"Total parameters: {total_parameters}")
    print(f"Total operations: {total_operations}")

    # 绘制每层的参数和操作次数折线图
    layers = [layer[0] for layer in model.named_modules() if isinstance(layer[1], (nn.Linear, nn.ReLU))]
    layer_parameters = [layer[1].weight.numel() for layer in model.named_modules() if isinstance(layer[1], (nn.Linear, nn.ReLU))]
    layer_operations = [layer[1].weight.numel() * layer[1].bias.numel() for layer in model.named_modules() if isinstance(layer[1], nn.Linear)]

    plt.figure(figsize=(10, 5))
    plt.plot(layers, layer_parameters, label='Parameters')
    plt.plot(layers, layer_operations, label='Operations')
    plt.xlabel('Layer Index')
    plt.ylabel('Complexity (O(n))')
    plt.legend()
    plt.title('Complexity of Each Layer in CombinedModel')
    plt.xticks(range(len(layers)), layers, rotation=90)
    plt.tight_layout()
    plt.show()

# 设置参数
graph_feat_size = 1024
input_size = 1024
hidden_size = 512
num_classes = 3
num_layers = 5

# 调用函数可视化模型复杂度
visualize_combined_model_complexity(graph_feat_size, input_size, hidden_size, num_classes, num_layers)
