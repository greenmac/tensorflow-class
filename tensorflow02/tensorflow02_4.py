# tensorflow簡單範例*****
import tensorflow as tf
import numpy as np

# 使用numpy生成100個隨機點
x_data = np.random.rand(100)
y_data = x_data*0.1 + 0.2

# 創造一個線性模型
# b = tf.Variable(0.)
# k = tf.Variable(0.)

b = tf.Variable(1.1)
k = tf.Variable(0.5)

y = k*x_data + b

# 二次代價函數
loss = tf.reduce_mean(tf.square(y_data-y)) # reduce_mean求平均值的意思,square是求平方()
# 定義一個梯度下降來進行訓練的優化器
optimizer = tf.train.GradientDescentOptimizer(0.2) # 學習率怎麼定義的?
# 最小化代價函數
train = optimizer.minimize(loss)

init = tf.global_variables_initializer() # 初始化變量

with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        if (step % 20) == 0:
            print(step, sess.run([k, b]))