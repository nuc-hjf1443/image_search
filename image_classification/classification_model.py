import torch
import torch.nn as nn

# 自定义基于CNN的分类器
class Classifier(nn.Module):
    def __init__(self, n_classes=5):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            # 第一层卷积-池化
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 第二层卷积-池化
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            # 全连接层
            nn.Linear(4096, n_classes)
        )

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    input = torch.randn(10, 3, 64, 64)
    model = Classifier()
    output = model(input)
    print(output.shape)