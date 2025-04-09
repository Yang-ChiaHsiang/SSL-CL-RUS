# +
import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn

def consistency_cost(model, teacher_model, imgs, p_masks):
    # 确保模型在训练模式，以便参数更新正常进行
    model.train()
    teacher_model.eval()  # 教师模型在评估模式，不更新参数

    # 禁用对比学习（如果适用）
    model.module.contrast = False
    teacher_model.module.contrast = False

    # 获取学生模型的输出
    output1 = model(imgs)  # [batch_size, num_classes, height, width]

    # 获取教师模型的输出，并确保不计算梯度
    with torch.no_grad():
        output2 = teacher_model(imgs)

    # 根据任务需求，选择合适的激活函数
    # 对于多类别分类任务，使用 softmax
    output1_probs = F.softmax(output1, dim=1)
    output2_probs = F.softmax(output2, dim=1)

    # 对于二分类任务，使用 sigmoid
    # output1_probs = torch.sigmoid(output1)
    # output2_probs = torch.sigmoid(output2)

    # 计算 MSE 损失
    loss = F.mse_loss(output1_probs, output2_probs)

    return loss
