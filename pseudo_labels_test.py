import torch
import numpy as np
import pandas as pd

"""
input:
    c：[batch_size, num_classes] 每一行代表一个样本对各个类别的预测概率
    pseudo_label_cur：[batch_size] 样本的当前伪标签（上一轮生成）
    index:[batch_size] 当前批次样本的索引
output:
    pseudo_label_nxt:[batch_size] 更新后的伪标签
    index:未经修改的、传入函数的样本索引
"""
def generate_pseudo_labels(c, pseudo_label_cur, index, alpha, gamma, cluster_num):
    batch_size = c.shape[0]
    device = c.device
    pseudo_label_nxt = -torch.ones(batch_size, dtype=torch.long).to(device)  # 初始化一个全为 -1（表示未标记）的伪标签向量
    tmp = torch.arange(0, batch_size).to(device)  # 用来引用样本的原始位置

    prediction = c.argmax(dim=1)  # 每个样本最可能属于的类别 [batch_size]
    confidence = c.max(dim=1).values  # 对应的置信度
    unconfident_pred_index = confidence < alpha  # 比较每个样本的置信度与预设的阈值 不可信则为True
    pseudo_per_class = np.ceil(batch_size / cluster_num * gamma).astype(int)  # 计算每个类别应该分配的伪标签数量 np.ceil 计算结果为小数向上取整
    for i in range(cluster_num):
        class_idx = prediction == i  # prediction 中预测类别为 i 的样本为True
        if class_idx.sum() == 0:
            continue
        confidence_class = confidence[class_idx]  # 从 confidence 数组中选择出预测为类别 i 的样本的置信度
        num = min(confidence_class.shape[0], pseudo_per_class)
        confident_idx = torch.argsort(-confidence_class)  # 降序排序 得到的confident_idx为预测为类别 i的样本 排序后的置信度索引
        for j in range(num):  # 遍历排序后的前 num 个最高置信度的样本
            # tmp[class_idx] 返回属于第 i 类的样本的原始批次索引
            # confident_idx[j] 返回经过排序的 confidence_class 中置信度第 j 高的样本的索引 指向的是筛选后数组中的位置，而不是整个原始批次
            idx = tmp[class_idx][confident_idx[j]]
            pseudo_label_nxt[idx] = i  # 将选中样本的伪标签设置为其预测类别 i

    todo_index = pseudo_label_cur == -1  # 尚未被标记的样本置为-1
    pseudo_label_cur[todo_index] = pseudo_label_nxt[todo_index]  # 更新未标记的样本的伪标签
    pseudo_label_nxt = pseudo_label_cur
    pseudo_label_nxt[unconfident_pred_index] = -1
    return pseudo_label_nxt.cpu(), index


# Load the Excel file
file_path = './p_values.csv'
c = pd.read_csv(file_path)
c = torch.from_numpy(c.values)

cluster_num = 37
alpha=0.9
gamma=0.5

pseudo_labels = -torch.ones(64, dtype=torch.long)#有用
index = torch.tensor(np.arange(64))
pseudo_label_nxt, index = generate_pseudo_labels(c, pseudo_labels, index, alpha, gamma, cluster_num)

print(f"pseudo_label_nxt:\n{pseudo_label_nxt}")
print(f"index:{index}")