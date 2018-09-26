# 05-1.第四周作業
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 載入數據集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

batch_size = 100 # 每個批次的大小(step-1 可更改數據的地方)
n_batch = mnist.train.num_examples // batch_size # 計算一共有多少個批次

# 定義兩個placeholder, 784=28*28(行*列)
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
lr = tf.Variable(0.001, dtype = tf.float32) # 學習率變量

# 創建一個簡單的神經網路
W1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1)) # (step-2 可更改數據的地方,神經元增減)
b1 = tf.Variable(tf.zeros([500]) + 0.1) # (step-3 可更改數據的地方,初始化的方式改變)
L1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
L1_drop = tf.nn.dropout(L1, keep_prob)

W2 = tf.Variable(tf.truncated_normal([500, 300], stddev=0.1)) # (step-2 可更改數據的地方,神經元增減)
b2 = tf.Variable(tf.zeros([300]) + 0.1) # (step-3 可更改數據的地方,初始化的方式改變)
L2 = tf.nn.tanh(tf.matmul(L1_drop, W2) + b2)
L2_drop = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.truncated_normal([300, 10], stddev=0.1)) # (step-2 可更改數據的地方,神經元增減)
b3 = tf.Variable(tf.zeros([10]) + 0.1) # (step-3 可更改數據的地方,初始化的方式改變)
prediction = tf.nn.softmax(tf.matmul(L2_drop, W3) + b3)

# 交叉熵代價函數
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
# 訓練
train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

# 初始化變量
init = tf.global_variables_initializer()

# 結果是存放在一個布爾型列表中
# tf.equal(A, B)是对比这两个矩阵或者向量的相等的元素，如果是相等的那就返回True，反正返回False，返回的值的矩阵维度和A是一样的
# tf.argmax(vector, 1)：返回的是vector中的最大值的索引号，如果vector是一个向量，那就返回一个值，如果是一个矩阵，那就返回一个向量，这个向量的每一个维度都是相对应矩阵行的最大值元素的索引号
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
# 求準確率
# tf.cast()轉化類型,布爾型會變 1 or 0
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session(config=tf.ConfigProto(device_count={'gpu':0})) as sess:
    sess.run(init)
    for epoch in range(51): # (step-4 可更改數據的地方)
        sess.run(tf.assign(lr, 0.001 * (0.95 ** epoch)))
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys, keep_prob:1.0}) # 如果keep_prob:0.7,表示上面的神經元只有70%在工作

        learning_rate = sess.run(lr)
        acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels, keep_prob:1.0})
        print('Iter ' + str(epoch) + ',Testing Accuracy= ' + str(acc) + ',Learning Rate= ' + str(learning_rate))

# 訓練的數據
# Iter 0,Testing Accuracy= 0.8352,Learning Rate= 0.001
# Iter 1,Testing Accuracy= 0.8405,Learning Rate= 0.00095
# Iter 2,Testing Accuracy= 0.8317,Learning Rate= 0.0009025
# Iter 3,Testing Accuracy= 0.8293,Learning Rate= 0.000857375
# Iter 4,Testing Accuracy= 0.8269,Learning Rate= 0.00081450626
# Iter 5,Testing Accuracy= 0.8411,Learning Rate= 0.0007737809
# Iter 6,Testing Accuracy= 0.8356,Learning Rate= 0.0007350919
# Iter 7,Testing Accuracy= 0.8287,Learning Rate= 0.0006983373
# Iter 8,Testing Accuracy= 0.8266,Learning Rate= 0.0006634204
# Iter 9,Testing Accuracy= 0.8301,Learning Rate= 0.0006302494
# Iter 10,Testing Accuracy= 0.8369,Learning Rate= 0.0005987369
# Iter 11,Testing Accuracy= 0.8301,Learning Rate= 0.0005688001
# Iter 12,Testing Accuracy= 0.8333,Learning Rate= 0.0005403601
# Iter 13,Testing Accuracy= 0.8325,Learning Rate= 0.0005133421
# Iter 14,Testing Accuracy= 0.8321,Learning Rate= 0.000487675
# Iter 15,Testing Accuracy= 0.824,Learning Rate= 0.00046329122
# Iter 16,Testing Accuracy= 0.8339,Learning Rate= 0.00044012666
# Iter 17,Testing Accuracy= 0.8191,Learning Rate= 0.00041812033
# Iter 18,Testing Accuracy= 0.8306,Learning Rate= 0.00039721432
# Iter 19,Testing Accuracy= 0.8306,Learning Rate= 0.0003773536
# Iter 20,Testing Accuracy= 0.8215,Learning Rate= 0.00035848594
# Iter 21,Testing Accuracy= 0.827,Learning Rate= 0.00034056162
# Iter 22,Testing Accuracy= 0.8301,Learning Rate= 0.00032353355
# Iter 23,Testing Accuracy= 0.8301,Learning Rate= 0.00030735688
# Iter 24,Testing Accuracy= 0.8242,Learning Rate= 0.000291989
# Iter 25,Testing Accuracy= 0.8261,Learning Rate= 0.00027738957
# Iter 26,Testing Accuracy= 0.8342,Learning Rate= 0.0002635201
# Iter 27,Testing Accuracy= 0.8229,Learning Rate= 0.00025034408
# Iter 28,Testing Accuracy= 0.824,Learning Rate= 0.00023782688
# Iter 29,Testing Accuracy= 0.8263,Learning Rate= 0.00022593554
# Iter 30,Testing Accuracy= 0.8284,Learning Rate= 0.00021463877
# Iter 31,Testing Accuracy= 0.8276,Learning Rate= 0.00020390682
# Iter 32,Testing Accuracy= 0.8361,Learning Rate= 0.00019371149
# Iter 33,Testing Accuracy= 0.8245,Learning Rate= 0.0001840259
# Iter 34,Testing Accuracy= 0.8338,Learning Rate= 0.00017482461
# Iter 35,Testing Accuracy= 0.8368,Learning Rate= 0.00016608338
# Iter 36,Testing Accuracy= 0.8274,Learning Rate= 0.00015777921
# Iter 37,Testing Accuracy= 0.8336,Learning Rate= 0.00014989026
# Iter 38,Testing Accuracy= 0.8292,Learning Rate= 0.00014239574
# Iter 39,Testing Accuracy= 0.8292,Learning Rate= 0.00013527596
# Iter 40,Testing Accuracy= 0.8349,Learning Rate= 0.00012851215
# Iter 41,Testing Accuracy= 0.8268,Learning Rate= 0.00012208655
# Iter 42,Testing Accuracy= 0.8321,Learning Rate= 0.00011598222
# Iter 43,Testing Accuracy= 0.8411,Learning Rate= 0.00011018311
# Iter 44,Testing Accuracy= 0.8294,Learning Rate= 0.000104673956
# Iter 45,Testing Accuracy= 0.8201,Learning Rate= 9.944026e-05
# Iter 46,Testing Accuracy= 0.8242,Learning Rate= 9.446825e-05
# Iter 47,Testing Accuracy= 0.819,Learning Rate= 8.974483e-05
# Iter 48,Testing Accuracy= 0.8364,Learning Rate= 8.525759e-05
# Iter 49,Testing Accuracy= 0.8218,Learning Rate= 8.099471e-05
# Iter 50,Testing Accuracy= 0.8227,Learning Rate= 7.6944976e-05