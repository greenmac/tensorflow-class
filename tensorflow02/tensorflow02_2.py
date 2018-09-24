# 02_2.變量
import tensorflow as tf

# 01
# x = tf.Variable([1, 2])
# a = tf.constant([3, 3])
# sub = tf.subtract(x, a) # 增加一個減法op
# add = tf.add(x, sub) # 增加一個加法op
# init = tf.global_variables_initializer() # 初始化變量
# 
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(sub))
#     print(sess.run(add))

# 02
state = tf.Variable(0, name='counter') # 創建一個變量初始化為0
new_value = tf.add(state, 1) # 創建一個op, 作用是使state加1
update = tf.assign(state, new_value) # 賦值op
init = tf.global_variables_initializer() # 初始化變量

with tf.Session() as sess:
    sess.run(init)
    # print(sess.run(state))
    for _ in range(5):
        sess.run(update)
        print(sess.run(state))