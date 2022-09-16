# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 20:00:14 2019

@author: xiuzhang
"""

# -*- coding: utf-8 -*-
"""
Created on 2019-11-30 下午6点 写于武汉大学

@author: Eastmount CSDN YXZ
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3 #权重0.1 偏置0.3

#------------------开始创建tensorflow结构------------------
# 权重和偏置
Weights = tf.Variable(tf.random.uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

# # 预测值y
# y = Weights * x_data + biases
#
# # 损失函数
# loss = tf.reduce_mean(tf.square(y-y_data))
def cost():
    y_model = Weights * x_data+ biases
    error = tf.reduce_mean(tf.square(y_data - y_model))
    return error
# 建立神经网络优化器
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5) #学习效率
optimizer = tf.optimizers.SGD(learning_rate=0.5)
cost_value = cost()
train = optimizer.minimize(cost, var_list=[Weights, biases], tape=tf.GradientTape())

# 初始化变量
init = tf.compat.v1.global_variables_initializer()
#------------------结束创建tensorflow结构------------------

# 可视化
plt.plot(x_data, y_data,'ro', marker='^', c='blue',label='original_data')
plt.legend()
plt.show()


# 定义Session
sess = tf.compat.v1.Session()

# 运行时Session就像一个指针 指向要处理的位置并激活
sess.run(init)

# 训练并且每隔20次输出结果
k = 0
for n in range(200):
    sess.run(train)
    if n % 20 == 0:
        print(n, sess.run(Weights), sess.run(biases))
        pre = x_data * sess.run(Weights) + sess.run(biases)

        # 可视化分析
        k = k + 1
        plt.subplot(5, 2, k)
        plt.plot(x_data, y_data, 'ro')
        plt.plot(x_data, pre, label=k)
        plt.legend()

plt.show()
