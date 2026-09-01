import torch


# 训练一个轮次，返回平均训练损失
def train_epoch(model, device, train_loader, loss_fn, optimizer):
    model.train()
    train_loss = 0.0
    for input, target in train_loader:
        input = input.to(device)
        target = target.to(device)
        # 前向传播
        output = model(input)
        # 计算损失
        loss = loss_fn(output, target)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 梯度清零
        optimizer.zero_grad()
        # 累加损失
        train_loss += loss.item()
    # 返回本轮平均损失
    return train_loss / len(train_loader)


# 训练一个轮次，返回平均验证损失
def val_step(model, device, val_loader, loss_fn):
    model.eval()
    val_loss = 0.0
    val_dataset_num = 0 # 累加验证集数据总数
    with torch.no_grad():
        for input, target in val_loader:
            input = input.to(device)
            target = target.to(device)
            # 前向传播
            output = model(input)
            # 计算损失
            loss = loss_fn(output, target)
            # 累加损失
            val_loss += loss.item() * input.shape[0]
            val_dataset_num += input.shape[0]
    # 返回本轮平均损失
    return val_loss / val_dataset_num


# 训练一个轮次，返回评价指标-准确率
def test_step(model, device, test_loader):
    model.eval()
    test_correct_num = 0
    test_dataset_num = 0
    with torch.no_grad():
        for input, target in test_loader:
            input = input.to(device)
            target = target.to(device)
            # 前向传播
            output = model(input)
            # 转换成预测分类
            pred = output.argmax(dim=1)
            # 累加预测准确数量，以及总数量
            test_correct_num += pred.eq(target).sum().item()
            test_dataset_num += input.shape[0]
    # 返回本轮平均损失
    return test_correct_num / test_dataset_num
