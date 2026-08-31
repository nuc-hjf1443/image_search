import torch
from PIL import Image
import torchvision.transforms as T
import os
from torch.utils.data import Dataset, random_split, DataLoader

from denoising_config import *
from common.utils import sorted_alphanum


# 自定义数据集类
class NoisyImageDataset(Dataset):
    # 初始化：传入图片根目录，以及预处理转换操作
    def __init__(self, main_dir, transform=None):
        self.main_dir = main_dir
        self.transform = transform
        self.img_names = sorted_alphanum(os.listdir(main_dir))

    # 获取数据集大小
    def __len__(self):
        return len(self.img_names)

    # 根据索引号得到（input, target）
    def __getitem__(self, idx):
        # 1. 根据索引号找到文件名，读取图片数据
        img_path = os.path.join(self.main_dir, self.img_names[idx])
        img = Image.open(img_path).convert('RGB')
        # 2. 将原始图片转换为符合模型输入要求的张量,这就是重构图像的目标
        if self.transform is not None:
            img_original_tensor = self.transform(img)
        else:
            raise ValueError("Transform must be provided!")
        # 3. 添加随机噪声（高斯噪声），构建输入数据
        img_noise_tensor = img_original_tensor + torch.randn_like(img_original_tensor) * NOISE_FACTOR
        img_noise_tensor = img_noise_tensor.clamp(0., 1.)
        # 将输入和目标返回
        return img_noise_tensor, img_original_tensor

# 创建并切分数据集
def create_dataset():
    # 定义转换操作
    transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
    ])
    dataset = NoisyImageDataset(main_dir=IMG_PATH, transform=transform)
    # 划分数据集
    train_dataset, val_dataset, test_dataset = random_split(dataset, [TRAIN_RATIO, VAL_RATIO, TEST_RATIO])
    return train_dataset, val_dataset, test_dataset

if __name__ == '__main__':
    train_dataset, val_dataset, test_dataset = create_dataset()
    print(len(train_dataset))
    print(len(val_dataset))
    print(len(test_dataset))
