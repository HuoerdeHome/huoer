import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


# 加载预训练的ResNet50模型
model = models.resnet50(pretrained=True)
# torch.save(model.state_dict(), 'model_weights.pth')  # 保存模型权重
# 将模型设置为评估模式
model.eval()

# 定义图像预处理操作
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 获取文件夹中的所有图片和txt文件
folder_path = 'datatset'
image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]
txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

# 初始化结果列表
result_list = []
label_list = []

for file in os.listdir(folder_path):
    if file.endswith('.txt'):
        txt_path = os.path.join(folder_path, file)
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            label = lines[0].strip()  # 读取第一行作为标签并去除两端的空白字符
            label_list.append(label)
# 遍历所有图片和txt文件
for image_file in image_files:
    # 加载图像并进行预处理
    image_path = os.path.join(folder_path, image_file)
    image = Image.open(image_path)
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    # 使用GPU进行计算（如果可用）
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    # 提取图像特征
    with torch.no_grad():
        output = model(input_batch)
        # print(output.shape)

    # 读取对应的txt文件
    txt_file = image_file.replace('.jpg', '.txt').replace('.png', '.txt')
    txt_path = os.path.join(folder_path, txt_file)
    with open(txt_path, 'r',) as f:
        feature_str = f.read()

    # 将特征字符串转换为浮点数列表
    feature_list = [float(x) for x in feature_str.split()]

    # 将特征列表转换为NumPy数组
    feature_array = np.array(feature_list)

    # 将NumPy数组转换为PyTorch张量
    feature_tensor = torch.from_numpy(feature_array)
    feature_tensor = feature_tensor.unsqueeze(0)
    # print(feature_tensor.shape)

    # 将特征张量与图像特征向量拼接在一起
    combined_features = torch.cat((output, feature_tensor), dim=1)
    # print(combined_features.shape)

    # 将拼接后的特征向量添加到结果列表中
    result_list.append(combined_features)

print(label_list)
print(result_list)
