# 04-2.Dropout
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

# 創建一個簡單的神經網路
W1 = tf.Variable(tf.truncated_normal([784, 2000], stddev=0.1)) # (step-2 可更改數據的地方,神經元增減)
b1 = tf.Variable(tf.zeros([2000]) + 0.1) # (step-3 可更改數據的地方,初始化的方式改變)
L1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
L1_drop = tf.nn.dropout(L1, keep_prob)

W2 = tf.Variable(tf.truncated_normal([2000, 2000], stddev=0.1)) # (step-2 可更改數據的地方,神經元增減)
b2 = tf.Variable(tf.zeros([2000]) + 0.1) # (step-3 可更改數據的地方,初始化的方式改變)
L2 = tf.nn.tanh(tf.matmul(L1_drop, W2) + b2)
L2_drop = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.truncated_normal([2000, 1000], stddev=0.1)) # (step-2 可更改數據的地方,神經元增減)
b3 = tf.Variable(tf.zeros([1000]) + 0.1) # (step-3 可更改數據的地方,初始化的方式改變)
L3 = tf.nn.tanh(tf.matmul(L2_drop, W3) + b3)
L3_drop = tf.nn.dropout(L3, keep_prob)

W4 = tf.Variable(tf.truncated_normal([1000, 10], stddev=0.1)) # (step-2 可更改數據的地方,神經元增減)
b4 = tf.Variable(tf.zeros([10]) + 0.1) # (step-3 可更改數據的地方,初始化的方式改變)
prediction = tf.nn.softmax(tf.matmul(L3_drop, W4) + b4)

# 二次代價函數
# loss= tf.reduce_mean(tf.square(y-prediction))
# 交叉熵
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

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

with tf.Session(config=tf.ConfigProto(device_count={'gpu':0})) as sess:
    sess.run(init)
    for epoch in range(31): # (step-4 可更改數據的地方)
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys, keep_prob:1.0}) # 如果keep_prob:0.7,表示上面的神經元只有70%在工作

        test_acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels, keep_prob:1.0})
        train_acc = sess.run(accuracy, feed_dict={x:mnist.train.images, y:mnist.train.labels, keep_prob:1.0})
        print('Iter ' + str(epoch) + ',Testing Accuracy ' + str(test_acc) + ',Training Accuracy ' + str(train_acc))


# 明顯 Dropout(keep_prob:1.0) 比 Dropout(keep_prob:0.7) 較快達到效果,100% vs 70% 的效率

# 訓練的數據(Dropout(keep_prob:1.0)))
# Iter 0,Testing Accuracy 0.9494,Training Accuracy 0.95805454
# Iter 1,Testing Accuracy 0.9582,Training Accuracy 0.97507274
# Iter 2,Testing Accuracy 0.9641,Training Accuracy 0.98205453
# Iter 3,Testing Accuracy 0.9639,Training Accuracy 0.98585457
# Iter 4,Testing Accuracy 0.9682,Training Accuracy 0.9879818
# Iter 5,Testing Accuracy 0.9678,Training Accuracy 0.9895091
# Iter 6,Testing Accuracy 0.9691,Training Accuracy 0.99061817
# Iter 7,Testing Accuracy 0.97,Training Accuracy 0.9914727
# Iter 8,Testing Accuracy 0.969,Training Accuracy 0.9921455
# Iter 9,Testing Accuracy 0.9699,Training Accuracy 0.99252725
# Iter 10,Testing Accuracy 0.9695,Training Accuracy 0.9929091
# Iter 11,Testing Accuracy 0.9698,Training Accuracy 0.99341816
# Iter 12,Testing Accuracy 0.9701,Training Accuracy 0.9936727
# Iter 13,Testing Accuracy 0.9703,Training Accuracy 0.99396366
# Iter 14,Testing Accuracy 0.9699,Training Accuracy 0.99414545
# Iter 15,Testing Accuracy 0.9702,Training Accuracy 0.99436367
# Iter 16,Testing Accuracy 0.9705,Training Accuracy 0.9944909
# Iter 17,Testing Accuracy 0.971,Training Accuracy 0.99474543
# Iter 18,Testing Accuracy 0.9716,Training Accuracy 0.99487275
# Iter 19,Testing Accuracy 0.9718,Training Accuracy 0.9949273
# Iter 20,Testing Accuracy 0.9716,Training Accuracy 0.99496365
# Iter 21,Testing Accuracy 0.9716,Training Accuracy 0.9951091
# Iter 22,Testing Accuracy 0.9713,Training Accuracy 0.99523634
# Iter 23,Testing Accuracy 0.9717,Training Accuracy 0.99529094
# Iter 24,Testing Accuracy 0.9718,Training Accuracy 0.99536365
# Iter 25,Testing Accuracy 0.9718,Training Accuracy 0.9954364
# Iter 26,Testing Accuracy 0.972,Training Accuracy 0.99552727
# Iter 27,Testing Accuracy 0.9719,Training Accuracy 0.9955636
# Iter 28,Testing Accuracy 0.9723,Training Accuracy 0.9956727
# Iter 29,Testing Accuracy 0.9721,Training Accuracy 0.9957455
# Iter 30,Testing Accuracy 0.9723,Training Accuracy 0.9958909

# 訓練的數據(Dropout(keep_prob:0.7)))
# Iter 0,Testing Accuracy 0.9187,Training Accuracy 0.9132909
# Iter 1,Testing Accuracy 0.9312,Training Accuracy 0.92634547
# Iter 2,Testing Accuracy 0.9364,Training Accuracy 0.9357273
# Iter 3,Testing Accuracy 0.9412,Training Accuracy 0.9401091
# Iter 4,Testing Accuracy 0.9423,Training Accuracy 0.9453091
# Iter 5,Testing Accuracy 0.9492,Training Accuracy 0.9499818
# Iter 6,Testing Accuracy 0.9499,Training Accuracy 0.95234543
# Iter 7,Testing Accuracy 0.9528,Training Accuracy 0.9552182
# Iter 8,Testing Accuracy 0.9532,Training Accuracy 0.9563091
# Iter 9,Testing Accuracy 0.9577,Training Accuracy 0.95896363
# Iter 10,Testing Accuracy 0.9578,Training Accuracy 0.96043634
# Iter 11,Testing Accuracy 0.958,Training Accuracy 0.96214545
# Iter 12,Testing Accuracy 0.9579,Training Accuracy 0.9634727
# Iter 13,Testing Accuracy 0.9593,Training Accuracy 0.9648909
# Iter 14,Testing Accuracy 0.9615,Training Accuracy 0.9664182
# Iter 15,Testing Accuracy 0.9621,Training Accuracy 0.96729094
# Iter 16,Testing Accuracy 0.9613,Training Accuracy 0.9678364
# Iter 17,Testing Accuracy 0.9623,Training Accuracy 0.96778184
# Iter 18,Testing Accuracy 0.9647,Training Accuracy 0.9697273
# Iter 19,Testing Accuracy 0.9644,Training Accuracy 0.9707091
# Iter 20,Testing Accuracy 0.9652,Training Accuracy 0.97074544
# Iter 21,Testing Accuracy 0.9645,Training Accuracy 0.9718182
# Iter 22,Testing Accuracy 0.9655,Training Accuracy 0.9722
# Iter 23,Testing Accuracy 0.9678,Training Accuracy 0.9737818
# Iter 24,Testing Accuracy 0.9678,Training Accuracy 0.974
# Iter 25,Testing Accuracy 0.9681,Training Accuracy 0.9750364
# Iter 26,Testing Accuracy 0.9687,Training Accuracy 0.9755818
# Iter 27,Testing Accuracy 0.9686,Training Accuracy 0.9764364
# Iter 28,Testing Accuracy 0.9701,Training Accuracy 0.97678185
# Iter 29,Testing Accuracy 0.9694,Training Accuracy 0.9774909
# Iter 30,Testing Accuracy 0.9702,Training Accuracy 0.97810906