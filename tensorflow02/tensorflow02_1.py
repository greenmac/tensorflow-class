# 02_1.創建圖,啟動圖
import tensorflow as tf

m1 = tf.constant([[3, 3]]) # 創建一個常量op
m2 = tf.constant([[2], [3]]) # 創建一個常量op
product =  tf.matmul(m1, m2) # 創建一個矩陣乘法op, 把m1和m2傳入
# print(product)

# sess = tf.Session() # 定義一個會話, 啟動默認圖
with tf.Session() as sess:
    result = sess.run(product) # 調用sess的run方法來執行矩陣乘法op,run(product)觸發了途中3個op 
    print(result)