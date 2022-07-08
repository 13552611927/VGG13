import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential


def preprocess(x,y):
    x = tf.cast(x,dtype=tf.float32)/255.
    y = tf.cast(y,dtype=tf.int32)
    return x,y
  
  
#import data from cifar100 datasets
(x,y),(x_test,y_test) = datasets.cifar100.load_data()
y = tf.squeeze(y, axis=1)
y_test = tf.squeeze(y_test, axis=1)
print(x.shape, y.shape, x_test.shape, y_test.shape)


#prepare data for training the network
train_db = tf.data.Dataset.from_tensor_slices((x,y))
train_db = train_db.shuffle(1000).map(preprocess).batch(64)


#prepare data for testing the network
test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_db = test_db.map(preprocess).batch(64)


#check the attributes of the data in training sets
sample = next(iter(train_db))
print('sample:',sample[0].shape,sample[1].shape,tf.reduce_min(sample[0]),tf.reduce_max(sample[0]))
