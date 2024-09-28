from PIL import Image
import torchvision.transforms as transforms
from train import model, num_classes, input_size, hidden_size  # 从train.py文件中导入训练好的模型
import torch
from model import FCNN
from dataset import result_list, label_list

# 将数据转换为PyTorch张量
test_data = torch.cat(result_list).float()
test_labels = [float(label) if isinstance(label, str) else label for label in label_list]
test_labels = torch.tensor(test_labels)

# 在测试集上评估模型性能
with torch.no_grad():
    test_outputs = model(test_data)
    _, test_predicted = torch.max(test_outputs, 1)
    test_accuracy = (test_predicted == test_labels).sum().item() / len(test_labels)
    print(f'Test Accuracy: {test_accuracy:.4f}')

# # 加载模型权重
model = FCNN(input_size, hidden_size, num_classes)
model.load_state_dict(torch.load('model_weights.pth'))

# 加载原始图片
image_path = 'datatset/黄芪 (10).jpg'
image = Image.open(image_path)

# 定义图像预处理操作
transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 将原始图片转换为PyTorch张量
image_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(image_tensor)
    _, predicted = torch.max(output, 1)
    predicted_index = torch.argmax(predicted).item()
    if 0 <= predicted_index < len(label_list):
        predicted_class = label_list[predicted_index]
        print("预测类别：", predicted_class)
    else:
        print("预测出错")
print('😀')
