import torchvision.models as models
import torch
from torch import nn
import time
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
    def __init__(self, input_size, num_classes):  # 初始化函数，接收输入层大小、隐藏层大小和输出层类别数作为参数
        super(FCNN, self).__init__()  # 调用父类的初始化函数
        self.fc1 = torch.nn.Linear(input_size, 512)  # 定义第一个全连接层，输入大小为1012，输出大小为512
        self.relu1 = torch.nn.ReLU()  # 定义第一个ReLU激活函数
        self.fc2 = torch.nn.Linear(512, 412)  # 定义第二个全连接层，输入大小为512，输出大小为512
        self.relu2 = torch.nn.ReLU()  # 定义第二个ReLU激活函数
        self.fc3 = torch.nn.Linear(412, 312)  # 定义第三个全连接层，输入大小为hidden_size，输出大小为hidden_size
        self.relu3 = torch.nn.ReLU()  # 定义第三个ReLU激活函数
        self.fc4 = torch.nn.Linear(312, 212)  # 定义第四个全连接层，输入大小为hidden_size，输出大小为hidden_size
        self.relu4 = torch.nn.ReLU()  # 定义第四个ReLU激活函数
        self.fc5 = torch.nn.Linear(212, num_classes)  # 定义第五个全连接层，输入大小为hidden_size，输出大小为num_classes

    def forward(self, x1, x2):  # 定义前向传播函数，接收两个输入x1和x2
        out = self.fc1(x1 + x2)  # 将输入x1和x2相加后传入第一个全连接层
        out = self.relu1(out)  # 对第一个全连接层的输出应用ReLU激活函数
        out = self.fc2(out)  # 将激活后的结果传入第二个全连接层
        out = self.relu2(out)  # 对第二个全连接层的输出应用ReLU激活函数
        out = self.fc3(out)  # 将激活后的结果传入第三个全连接层
        out = self.relu3(out)  # 对第三个全连接层的输出应用ReLU激活函数
        out = self.fc4(out)  # 将激活后的结果传入第四个全连接层
        out = self.relu4(out)  # 对第四个全连接层的输出应用ReLU激活函数
        out = self.fc5(out)  # 将激活后的结果传入第五个全连接层
        return out  # 返回第五个全连接层的输出结果


def measure_time(model, input_size):
    start_time = time.time()
    model(torch.randn(input_size))
    end_time = time.time()
    return end_time - start_time

def visualize_time_complexity(models, input_sizes):
    times = []
    for model in models:
        time_taken = measure_time(model, input_sizes[model.__class__.__name__])
        times.append(time_taken)

    labels = [model.__class__.__name__ for model in models]
    plt.bar(labels, times)
    plt.xlabel('Model')
    plt.ylabel('Time (seconds)')
    plt.title('Time Complexity of Models')
    plt.show()

if __name__ == "__main__":
    gshq_chn = GSHq_CHN(graph_feat_size=1012)
    fcnn = FCNN(input_size=1012, num_classes=3)

    models = {GSHq_CHN.__name__: gshq_chn, FCNN.__name__: fcnn}
    input_sizes = {GSHq_CHN.__name__: 1012, FCNN.__name__: 1012}

    # 添加一个示例输入
    HG1 = torch.randn(1, 1012)
    HG2 = torch.randn(1, 1012)

    # 修改measure_time函数以接受两个输入参数
    def measure_time(model, input1, input2):
        start_time = time.time()
        model(input1, input2)
        end_time = time.time()
        return end_time - start_time

    # 修改visualize_time_complexity函数以传递两个输入参数
    def visualize_time_complexity(models, input1, input2):
        times = []
        for model in models:
            time_taken = measure_time(model, input1, input2)
            times.append(time_taken)

        labels = [model.__class__.__name__ for model in models]
        plt.bar(labels, times)
        plt.xlabel('Model')
        plt.ylabel('Time (seconds)')
        plt.title('Time Complexity of Models')
        plt.show()

    visualize_time_complexity(models.values(), HG1, HG2)

