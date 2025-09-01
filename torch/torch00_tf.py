import tensorflow as tf
print('tensorflow version : ', tf.__version__)

if tf.config.list_physical_devices('GPI'):
    print('there is GPU')
else:
    print('NO GPU')
# cuda
cuda_version = tf.sysconfig.get_build_info()['cuda_version']
print(cuda_version)

# cudnn
cudnn_version = tf.sysconfig.get_build_info()['cudnn_version']
print(cuda_version)

