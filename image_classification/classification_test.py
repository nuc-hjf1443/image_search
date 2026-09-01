import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from common.utils import seed_everything
from classification_config import *
from classification_data import create_dataset
from classification_model import Classifier
from classification_engine import test_step


# 函数：测试一批数据的重构图像
def test_batch(model, test_loader, device):
    model.to(device)
    model.eval()
    # 1. 选取一批测试数据
    test_iter = iter(test_loader)
    images, labels = next(test_iter)
    # 2. 推理预测
    with torch.no_grad():
        images = images.to(device)
        # 前向传播
        outputs = model(images)
    print("输出形状：", outputs.shape)
    # 3. 将输出转换为分类预测标签
    pred_labels = outputs.argmax(dim=1).cpu().numpy()
    # 4. 将输入图像数据，方便画图
    images = images.permute(0, 2, 3, 1).cpu().numpy()

    # 画图
    fig, axes = plt.subplots(1, 10, figsize=(25, 4), sharex=True, sharey=True)
    for i in range(10):
        axes[i].imshow(images[i])
        axes[i].axis("off")
        # 打印真实标签
        print(f"label-{i + 1}: {labels[i]}")
        # 打印预测分类标签
        # 转换成中文分类
        pred_class = classification_names[pred_labels[i]]
        print(f"pred_label-{i + 1}: {pred_labels[i]}, 分类名：{pred_class}")
        print()

    plt.show()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(SEED)

    # 1. 创建数据集
    _, _, test_dataset = create_dataset()
    print("数据集创建完成！")

    # 2. 创建数据加载器
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE)
    print("数据加载器创建完成！")

    # 3. 定义模型
    classifier = Classifier()
    state_dict = torch.load(CLASSIFIER_MODEL_NAME, map_location=device)
    classifier.load_state_dict(state_dict)

    print("模型加载完成！")

    # 4. 测试
    print("测试结果如下：")
    test_batch(classifier, test_loader, device)

    test_acc = test_step(model=classifier, test_loader=test_loader, device=device)
    print("测试准确率：", test_acc)