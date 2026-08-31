import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from common.utils import seed_everything
from denoising_config import *
from denoising_data import create_dataset
from denoising_model import ConvDenoiser
from denoising_engine import test_step


# 函数：测试一批数据的重构图像
def test_batch(model, test_loader, device):
    model.to(device)
    model.eval()
    # 1. 选取一批测试数据
    test_iter = iter(test_loader)
    noisy_images, images = next(test_iter)
    # 2. 推理预测
    with torch.no_grad():
        noisy_images = noisy_images.to(device)
        # 前向传播
        outputs = model(noisy_images)
    print("输出重构图像形状：", outputs.shape)
    # 3. 数据转换，为画图做准备
    noisy_images = noisy_images.permute(0, 2, 3, 1).cpu().numpy()
    outputs = outputs.permute(0, 2, 3, 1).cpu().numpy()
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    # 4. 画图
    fig, axes = plt.subplots(nrows=3, ncols=10, figsize=(25, 4), sharex=True, sharey=True)
    for imgs, ax_row in zip([noisy_images, outputs, images], axes):
        for img, ax in zip(imgs, ax_row):
            ax.imshow(img)
            ax.set_axis_off()
    plt.show()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(SEED)

    # 1. 创建数据集
    _, _, test_dataset = create_dataset()
    print("数据集创建完成！")

    # 2. 创建数据加载器
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, drop_last=True)
    print("数据加载器创建完成！")

    # 3. 定义模型
    denoiser = ConvDenoiser()
    state_dict = torch.load(DENOISER_MODEL_NAME, map_location=device)
    denoiser.load_state_dict(state_dict)

    print("模型加载完成！")

    # 4. 测试
    print("测试结果如下：")
    test_batch(denoiser, test_loader, device)

    test_loss = test_step(model=denoiser, test_loader=test_loader, device=device, loss_fn=torch.nn.MSELoss())
    print("测试集均方误差：", test_loss)