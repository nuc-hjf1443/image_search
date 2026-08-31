import torch
import torch.nn as nn


# 自定义自编码器模型
class ConvDenoiser(nn.Module):
    def __init__(self):
        super(ConvDenoiser, self).__init__()
        # 编码器部分
        # 卷积层
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # 池化层(通用)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 解码器部分
        # 转置卷积层
        self.conv_t1 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.conv_t2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv_t3 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
        # 输出前普通卷积
        self.conv_out = nn.Conv2d(8, 3, kernel_size=3, padding=1)

    # 前向传播
    def forward(self, x):
        # 第一层卷积-池化
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        # 第二层卷积-池化
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        # 第三层卷积-池化
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        # print("encoded shape: ", x.shape)

        # 解码
        x = torch.relu(self.conv_t1(x))
        x = torch.relu(self.conv_t2(x))
        x = torch.relu(self.conv_t3(x))
        # 最后普通卷积
        x = torch.sigmoid(self.conv_out(x))
        return x


if __name__ == '__main__':
    input = torch.randn(10, 3, 64, 64)
    denoiser = ConvDenoiser()
    output = denoiser(input)
    print(output.shape)
