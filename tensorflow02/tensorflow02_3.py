# 03_3.fetch and feed
import tensorflow as tf

# fetch
# input1 = tf.constant(3.0)
# input2 = tf.constant(2.0)
# input3 = tf.constant(5.0)
# 
# add = tf.add(input2, input3)
# mu1 = tf.multiply(input1, add)
# 
# with tf.Session() as sess:
#     result = sess.run([mu1, add])
#     print(result)

# feed
# 創建占位符
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

with tf.Session() as sess:
    # feed的數據以字典的形式傳入
    print(sess.run(output, feed_dict={input1:[8.], input2:[2.]}))