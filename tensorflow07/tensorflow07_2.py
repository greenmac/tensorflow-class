# 07_2.遞歸神經網路RNN
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 載入數據集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 載入圖片是 28*28
n_inputs = 28 # 輸入一行, 一行有28個數據
max_time = 28 # 一共28行
lstm_size = 100 # 隱層單元
n_classes = 10 # 10個分類
batch_size = 50 # 每批次50個樣本
n_batch = mnist.train.num_examples // batch_size # 計算一共有多少個批次

# 這裡的none表示第一個維度可以是任意長度
x = tf.placeholder(tf.float32, [None, 784])
# 正確標籤
y = tf.placeholder(tf.float32, [None, 10])

# 初始化權值
weights = tf.Variable(tf.truncated_normal([lstm_size, n_classes], stddev=0.1))
# 初始化偏置值
biases = tf.Variable(tf.constant(0.1, shape=[n_classes]))

# 定義RNN網絡
def RNN(X, weights, biases):
    # input=[batch_siaze, max_time, n_inputs]
    inputs = tf.reshape(X, [-1, max_time, n_inputs])
    # 定義LSTM基本CELL
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    # final_state[0]是cell state
    # final_state[1]是hidden_state
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)
    results = tf.nn.softmax(tf.matmul(final_state[1], weights) + biases)
    return results

# 計算RNN的返回結果
prediction = RNN(x, weights, biases)
# 損失函數
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
# 使用AdamOptimizer進行優化
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 結果存放在一個布爾列表中
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1)) # argmax返回一維張量中最大的值所在的位置
# 求準確率, tf.cast：用于改变某个张量的数据类型
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # 把correct_prediction變為float32類型
# 初始化
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(6):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys})

        acc =sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
        print('Iter ' + str(epoch) + ', Testing Accuracy= ' + str(acc))

# 訓練數據(LSTM)
# Iter 0, Testing Accuracy= 0.7432
# Iter 1, Testing Accuracy= 0.811
# Iter 2, Testing Accuracy= 0.8322
# Iter 3, Testing Accuracy= 0.835
# Iter 4, Testing Accuracy= 0.9157
# Iter 5, Testing Accuracy= 0.9265