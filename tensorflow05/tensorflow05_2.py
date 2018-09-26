# 05-2.tensorboard網路結構
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 載入數據集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    
batch_size = 100 # 每個批次的大小(step-1 可更改數據的地方)
n_batch = mnist.train.num_examples // batch_size # 計算一共有多少個批次

# 命名空間
with tf.name_scope('input'):
    # 定義兩個placeholder, 784=28*28(行*列)
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')

with tf.name_scope('layer'):
    # 創建一個簡單的神經網路
    with tf.name_scope('wights'):
        W = tf.Variable(tf.zeros([784, 10])) # (step-2 可更改數據的地方,神經元增減)
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10])) # (step-3 可更改數據的地方,初始化的方式改變)
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x, W) + b
    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(wx_plus_b)

# 二次代價函數
with tf.name_scope('loss'):
    loss= tf.reduce_mean(tf.square(y-prediction))
with tf.name_scope('train'):
    # 使用梯度下降法
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss) # 學習率最小化
# 初始化變量
init = tf.global_variables_initializer()

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        # 結果是存放在一個布爾型列表中
        # tf.equal(A, B)是对比这两个矩阵或者向量的相等的元素，如果是相等的那就返回True，反正返回False，返回的值的矩阵维度和A是一样的
        # tf.argmax(vector, 1)：返回的是vector中的最大值的索引号，如果vector是一个向量，那就返回一个值，如果是一个矩阵，那就返回一个向量，这个向量的每一个维度都是相对应矩阵行的最大值元素的索引号
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    with tf.name_scope('accuracy'):
        # 求準確率
        # tf.cast()轉化類型,布爾型會變 1 or 0
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('logs/', sess.graph) # tf.summary.FileWriter() 將視覺化檔案輸出
    for epoch in range(1): # (step-4 可更改數據的地方)
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys})

        acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
        print('Iter ' + str(epoch) + ',Testing Accuracy ' + str(acc))
