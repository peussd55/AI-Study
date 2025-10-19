### <<15>>

import tensorflow as tf 
print(tf.__version__)   # 2.7.3

gpus = tf.config.list_physical_devices('GPU')
print(gpus)

if gpus:
    print('GPU 있다~')
else:
    print('GPU 없다~')
