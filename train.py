import paddle
import torch
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from dataset import result_list, label_list
from model import FCNN
from sklearn.metrics import recall_score
import matplotlib.colors as mcolors

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")  # 设置设备为GPU或CPU

# 划分训练集和验证集
train_data, val_data, train_labels, val_labels = train_test_split(result_list, label_list, test_size=0.2,
                                                                  random_state=42)

# 将数据转换为PyTorch张量
train_data = torch.cat(train_data).float()
val_data = torch.cat(val_data).float()

# 确保train_labels中的所有元素都是数值类型
train_labels = [float(label) if isinstance(label, str) else label for label in train_labels]
val_labels = [float(label) if isinstance(label, str) else label for label in val_labels]

train_labels = torch.tensor(train_labels)
val_labels = torch.tensor(val_labels)
# 创建TensorDataset
train_dataset = TensorDataset(train_data, train_labels)
val_dataset = TensorDataset(val_data, val_labels)

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 初始化模型、损失函数和优化器
input_size = train_data.shape[1]
print(input_size)

hidden_size = 512  # 隐藏层
num_classes = 3

model = FCNN(input_size, hidden_size, num_classes).to(device)
torch.save(model.state_dict(), 'model_weights.pth')  # 保存模型权重

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.01, step_size=5, gamma=0.2)  # 添加学习率衰减策略

num_epochs = 50
loss_list = []
accuracy_list = []
val_loss_list = []
val_accuracy_list = []
time_list = []  # 用于存储每个轮次的时间消耗
for epoch in range(num_epochs):
    start_time = time.time()  # 记录轮次开始时间
    outputs = model(train_data)
    train_labels = torch.tensor(train_labels).to(torch.long)
    loss = criterion(outputs, train_labels)
    loss_list.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == train_labels).sum().item() / len(train_labels)
    accuracy_list.append(accuracy)
    end_time = time.time()  # 记录轮次结束时间
    epoch_time = end_time - start_time  # 计算轮次时间消耗
    time_list.append(epoch_time)  # 将时间消耗添加到列表中
    print(
        f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}, Time: {epoch_time:.4f} seconds')

    # 在验证集上评估模型性能
    with torch.no_grad():
        val_outputs = model(val_data)
        torch.tensor(val_labels, dtype=torch.long)
        val_labels = torch.tensor(val_labels).to(torch.long)
        val_loss = criterion(val_outputs, val_labels)
        val_loss_list.append(val_loss.item())
        _, val_predicted = torch.max(val_outputs, 1)
        val_accuracy = (val_predicted == val_labels).sum().item() / len(val_labels)
        val_accuracy_list.append(val_accuracy)
        print(f'Validation Loss: {val_loss.item():.4f}, Validation Accuracy: {val_accuracy:.4f}')

# 计算平均时间复杂度
average_time = sum(time_list) / len(time_list)
print(f"Average Time per Epoch: {average_time:.4f} seconds")

# 可视化每个训练轮次的时间复杂度
plt.figure(figsize=(10, 6))

# 创建一个颜色列表，长度与time_list相同
colors = list(mcolors.TABLEAU_COLORS.values())[:len(time_list)]

plt.barh(range(1, num_epochs + 1), time_list, align='center', alpha=0.5, color=colors)
plt.xlabel('Time (seconds)')
plt.ylabel('Epoch')
plt.title('Training Time per Epoch')
plt.grid(True)
plt.show()
# 计算平均准确率
avg_accuracy = sum(accuracy_list) / len(accuracy_list)
print(f'Average Accuracy: {avg_accuracy:.4f}')

# 计算平均损失
avg_loss = sum(loss_list) / len(loss_list)
print(f'Average Loss: {avg_loss:.4f}')

# 计算召回率
# val_labels_np = val_labels.cpu().numpy()
# val_predicted_np = val_predicted.cpu().numpy()
recall = recall_score(val_labels.cpu(), val_predicted.cpu(), average='macro')
print(f'Recall: {recall:.4f}')

# 绘制损失和准确率折线图
# plt.figure()
# plt.plot(range(num_epochs), loss_list, label='Training Loss',marker='o')
# plt.plot(range(num_epochs), val_loss_list, label='Validation Loss',marker='o')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
#
# plt.figure()
# plt.plot(range(num_epochs), accuracy_list, label='Training Accuracy',marker='o')
# plt.plot(range(num_epochs), val_accuracy_list, label='Validation Accuracy',marker='o')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots(figsize=(8, 6))

# 绘制训练准确率
ax1.plot(range(num_epochs), accuracy_list, label='Training Accuracy', marker='o')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()

# 创建一个共享横轴的新坐标轴，并将其放在右侧
ax2 = ax1.twinx()

# 绘制训练损失
ax2.plot(range(num_epochs), loss_list, label='Training Loss', marker='o', color='r')
ax2.set_ylabel('Loss')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()
