# 05_4.tensorflow可視化*****有問題要修
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector

# 載入數據集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# 運行次數
max_step = 1001
# 圖片數量
image_num = 3000
# 文件路徑
# DIR = 'E:/python/tensorflow-class/'
DIR = 'E:/python/tensorflow-class/tensorflow05/'

# 定義會話
sess = tf.Session()

# 載入圖片
embedding = tf.Variable(tf.stack(mnist.test.images[:image_num]), trainable=False, name='embedding')

batch_size = 100 # 每個批次的大小(step-1 可更改數據的地方)
n_batch = mnist.train.num_examples // batch_size # 計算一共有多少個批次

# 參數概要
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('stddev', mean) # 平均值 # tf.summary.scalar 操作来分别输出学习速度和期望误差，可以给每个 scalary_summary 分配一个有意义的标签为 'learning rate' 和 'loss function'，执行后就可以看到可视化的图表
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean))) # tf.sqrt 計算元素的平方根
        tf.summary.scalar('stddev', stddev) # 標準差
        tf.summary.scalar('max', tf.reduce_max(var)) # 最大值
        tf.summary.scalar('min', tf.reduce_min(var)) # 最小值
        tf.summary.histogram('historgram', var) # 直方圖

# 命名空間
with tf.name_scope('input'):
    # 這裡的none表示第一個維度可以是任意長度
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    # 正確標籤
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')

# 顯示圖片
with tf.name_scope('input_reshape'):
    image_shpaed_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shpaed_input, 10)

with tf.name_scope('layer'):
    # 創建一個簡單的神經網路
    with tf.name_scope('wights'):
        W = tf.Variable(tf.zeros([784, 10])) # (step-2 可更改數據的地方,神經元增減)
        variable_summaries(W)
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10])) # (step-3 可更改數據的地方,初始化的方式改變)
        variable_summaries(b)
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x, W) + b
    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(wx_plus_b)

# 交叉熵
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    # 使用梯度下降法
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss) # 學習率最小化
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
        tf.summary.scalar('accuracy', accuracy)

# tf.gfile.Exists,確定路徑是否存在 
# 產生metadata文件
if tf.gfile.Exists(DIR + 'projector/projector/metadata.tsv'):
    tf.gfile.DeleteRecursively(DIR + 'projector/projector/metadata.tsv')
with open(DIR + 'projector/projector/metadata.tsv', 'w') as f:
    labels = sess.run(tf.argmax(mnist.test.labels[:], 1))
    for i in range(image_num):
        f.write(str(labels[i]) + '\n')

# 合併所有的summary
merged = tf.summary.merge_all()

projector_writer = tf.summary.FileWriter(DIR + 'projector/projector', sess.graph)
saver = tf.train.Saver()
config = projector.ProjectorConfig()
embed = config.embeddings.add()
embed.tensor_name = embedding.name
embed.metadata_path = DIR + 'projector/projector/metadata.tsv'
embed.sprite.image_path = DIR + 'projector/projector/mnist_10k_sprite.png'
embed.sprite.single_image_dim.extend([28, 28])
projector.visualize_embeddings(projector_writer, config)

for i in range(max_step):
    # 每次批次100個樣本
    batch_xs, batch_ys = mnist.train.next_batch(100)
    run_option = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    summary,_ = sess.run([merged, train_step], feed_dict={x:batch_xs, y:batch_ys}, options=run_option, run_metadata=run_metadata)
    projector_writer.add_run_metadata(run_metadata, 'step%03d' % i)
    projector_writer.add_summary(summary, i)
    if i * 100 == 0:
        acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
        print('Iter ' + str(i) + ',Testing Accuracy= ' + str(acc))

saver.save(sess, 'projector/projector/a_model.ckpt', global_step=max_steps)
projector_writer.close()
sess.close()