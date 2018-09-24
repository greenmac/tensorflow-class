# 03_2.MNIST數據及分類簡單版本(二次代價函數)
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 載入數據集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

batch_size = 100 # 每個批次的大小(step-1 可更改數據的地方)
n_batch = mnist.train.num_examples // batch_size # 計算一共有多少個批次

# 定義兩個placeholder, 784=28*28(行*列)
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 創建一個簡單的神經網路
W = tf.Variable(tf.zeros([784, 10])) # (step-2 可更改數據的地方,神經元增減)
b = tf.Variable(tf.zeros([10])) # (step-3 可更改數據的地方,初始化的方式改變)
prediction = tf.nn.softmax(tf.matmul(x, W) + b)

# 二次代價函數
loss= tf.reduce_mean(tf.square(y-prediction))
# 使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss) # 學習率最小化
# 初始化變量
init = tf.global_variables_initializer()

# 結果是存放在一個布爾型列表中
# tf.equal(A, B)是对比这两个矩阵或者向量的相等的元素，如果是相等的那就返回True，反正返回False，返回的值的矩阵维度和A是一样的
# tf.argmax(vector, 1)：返回的是vector中的最大值的索引号，如果vector是一个向量，那就返回一个值，如果是一个矩阵，那就返回一个向量，这个向量的每一个维度都是相对应矩阵行的最大值元素的索引号
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
# 求準確率
# tf.cast()轉化類型,布爾型會變 1 or 0
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21): # (step-4 可更改數據的地方)
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys})

        acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
        print('Iter ' + str(epoch) + ',Testing Accuracy ' + str(acc))

# 訓練的數據(二次代價函數)
# Iter 0,Testing Accuracy 0.8313
# Iter 1,Testing Accuracy 0.8715
# Iter 2,Testing Accuracy 0.8809
# Iter 3,Testing Accuracy 0.888
# Iter 4,Testing Accuracy 0.8935
# Iter 5,Testing Accuracy 0.8975
# Iter 6,Testing Accuracy 0.8993
# Iter 7,Testing Accuracy 0.9027
# Iter 8,Testing Accuracy 0.9038
# Iter 9,Testing Accuracy 0.9051
# Iter 10,Testing Accuracy 0.9065
# Iter 11,Testing Accuracy 0.9062
# Iter 12,Testing Accuracy 0.9086
# Iter 13,Testing Accuracy 0.9092
# Iter 14,Testing Accuracy 0.9102 # 比交叉熵慢達到目標
# Iter 15,Testing Accuracy 0.9105
# Iter 16,Testing Accuracy 0.9113
# Iter 17,Testing Accuracy 0.9121
# Iter 18,Testing Accuracy 0.9123
# Iter 19,Testing Accuracy 0.9135
# Iter 20,Testing Accuracy 0.9139