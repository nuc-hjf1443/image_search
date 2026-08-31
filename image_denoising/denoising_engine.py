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
    with torch.no_grad():
        for input, target in val_loader:
            input = input.to(device)
            target = target.to(device)
            # 前向传播
            output = model(input)
            # 计算损失
            loss = loss_fn(output, target)
            # 累加损失
            val_loss += loss.item()
    # 返回本轮平均损失
    return val_loss / len(val_loader)


# 训练一个轮次，返回评价指标-平均测试误差
def test_step(model, device, test_loader, loss_fn):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for input, target in test_loader:
            input = input.to(device)
            target = target.to(device)
            # 前向传播
            output = model(input)
            # 计算损失
            loss = loss_fn(output, target)
            # 累加损失
            test_loss += loss.item()
    # 返回本轮平均损失
    return test_loss / len(test_loader)
