# coding=UTF-8
'''
@Description: 线性回归，gluon实现
@Version: 
@Author: liguoying
@Date: 2019-04-22 10:30:12
'''

from mxnet import autograd, nd
from mxnet import gluon, init
from mxnet.gluon import loss as gloss
from mxnet.gluon import data as gdata
from mxnet.gluon import nn

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0] * features[:,0] + true_w[1] * features[:,1] + true_b
# 加入随机扰动
labels += nd.random.normal(scale=0.01, shape=labels.shape)

batch_size = 10
# 将训练数据的特征和标签组合
dataset = gdata.ArrayDataset(features, labels)
# 随机读取小批量
data_iter = gdata.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# for X, y in data_iter:
#     print(X, y)
#     break

# 定义模型
net = nn.Sequential()
# 线性回归只有输入层和输出层，没有中间隐藏层
net.add(nn.Dense(1))
# 初始化模型参数
net.initialize(init.Normal(sigma=0.01))
# 损失函数
loss = gloss.L2Loss()   # 平方损失又称L2损失
# 定义优化算法
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.03})
# 训练模型
num_epochs = 3
for epoch in range(1, num_epochs+1):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print("epoch: %d, loss: %f" % (epoch, l.mean().asnumpy()))

# print(net[0])
dense = net[0]
# 对比结果
print("True weight: %s, Predicted weight: %s" % (true_w, dense.weight.data()))
print("True bias: %s, Predicted bias: %s" % (true_b, dense.bias.data()))