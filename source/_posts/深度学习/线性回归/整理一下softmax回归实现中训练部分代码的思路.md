---
title: 整理一下softmax回归实现中训练部分代码的思路
tags:
  - 深度学习，线性回归
mathjax: "true"
date: 2026-02-08 13:00:00
---
<style>
/* 强制让 MathJax 公式容器支持横向滚动 */
.mjx-container, .MathJax_Display, .MathJax {
    overflow-x: auto !important; /* 超出宽度时显示滚动条 */
    overflow-y: hidden;          /* 隐藏垂直滚动条 */
    max-width: 100%;             /* 限制最大宽度为屏幕宽度 */
    -webkit-overflow-scrolling: touch; /* 优化移动端滑动体验 */
}
</style>

在训练前，我们需要利用一些函数来评估模型的分类精度
首先，如果`y_hat`是矩阵，那么假定第二个维度存储每个类的预测分数。 我们使用`argmax`获得每行中最大元素的索引来获得预测类别。 然后我们将预测类别与真实`y`元素进行比较。 由于等式运算符“`==`”对数据类型很敏感， 因此我们将`y_hat`的数据类型转换为与`y`的数据类型一致。 结果是一个包含0（错）和1（对）的张量。 最后，我们求和会得到正确预测的数量。
```python
def accuracy(y_hat, y): 
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1: # 如果是一个矩阵
        y_hat = y_hat.argmax(axis=1) # 获取每一行的最大值
    cmp = y_hat.astype(y.dtype) == y # 判断最大值是否与y中的每一行真实的最大值相同
    return float(cmp.astype(y.dtype).sum()) # 返回一个正确预测的数量
```
同样，对于任意数据迭代器data_iter可访问的数据集，我们可以评估在任意模型net的精度
```python
def evaluate_accuracy(net, data_iter): 
    """计算在指定数据集上模型的精度"""
    metric = Accumulator(2)  # 正确预测数、预测总数
    for X, y in data_iter: # 在迭代器中获取数据
        metric.add(accuracy(net(X), y), d2l.size(y)) # 计算正确数量
    return metric[0] / metric[1] # 返回精度
```
这里我们定义了一个类Accmulator，用于对多个变量进行累加
```python
class Accumulator: 
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n # 在声明实例时n有多少就代表着他想返回的精度的条例有多少
    def add(self, *args):
	    # 用于对列表中的每一项加上（正确数量，数据总量）
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0] * len(self.data)
    def __getitem__(self, idx):
	    # 定义了__getitem__方法说明了这个类创建的实例可以被索引化
        return self.data[idx]
```

## 训练
我们先考虑在一个轮次中我们是怎么训练的，这一切还是在代码里用注释讲清楚比较好
```python
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期"""
    # 将模型设置为训练模式 设置为训练模式时 模型会自动计算梯度
    if isinstance(net, torch.nn.Module):
        net.train()
    # 此时metric.add()方法能返回的是训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        # 计算损失
        l = loss(y_hat, y)
        # 在这里开始我们根据优化器的情况分成两种情况
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad() # 梯度清空 不然梯度就会被累加
            l.mean().backward() # 计算平均梯度
            updater.step() # 自动管理 net.parameters()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward() # 计算总梯度
            updater(X.shape[0]) # 把总梯度平均化
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]
```
训练的完整实现
```python
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型"""
    '''
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    '''
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater) # 训练一次
        test_acc = evaluate_accuracy(net, test_iter) # 测试集的精度
        '''animator.add(epoch + 1, train_metrics + (test_acc,))'''
    train_loss, train_acc = train_metrics # 返回训练集loss和精度
    # 如果不符合要求 则报错并抛出后面的值
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
```
