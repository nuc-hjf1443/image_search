import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from common.utils import *
from classification_config import *
from classification_data import create_dataset
from classification_model import Classifier
from classification_engine import *

from tqdm import tqdm

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed_everything(SEED)

# 1. 创建数据集
train_dataset, val_dataset, _ = create_dataset()
print("数据集创建完成！")

# 2. 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE)
print("数据加载器创建完成！")

#3. 定义模型
classifier = Classifier()
classifier.to(device)

# 4. 定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(classifier.parameters(), lr=LEARNING_RATE)

# 5. 训练模型
# 保存最小验证损失
min_val_loss = float("inf")
for epoch in tqdm(range(EPOCHS)):
    # 执行训练一个轮次
    train_loss = train_epoch(model=classifier, device=device, train_loader=train_loader, loss_fn=loss_fn, optimizer=optimizer)
    # 执行验证
    val_loss = val_step(model=classifier, device=device, val_loader=val_loader, loss_fn=loss_fn)
    print(f"\nEpoch: {epoch + 1} / {EPOCHS}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    # 判断如果验证损失下降就保存模型
    if val_loss < min_val_loss:
        print("验证集损失下降，保存模型...")
        min_val_loss = val_loss
        torch.save(classifier.state_dict(), CLASSIFIER_MODEL_NAME)
    else:
        print("验证集损失没有减小，不保存模型！")

print("训练完成！")