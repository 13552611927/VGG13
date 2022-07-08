network = Sequential([
    '''
    ~1 processing unit contains 2 Convulutional layers + 1 MaxPooling layers;
    ~the last unit contain 2 layers of 512 kernels instead of 2 layers of 1024 kernels because 512 kernels would have less params,
    which is easier for the GPUs to process.
    '''
    layers.Conv2D(64, kernel_size = [3, 3], padding = 'same', activation = tf.nn.relu),
    layers.Conv2D(64, kernel_size = [3, 3], padding = 'same', activation = tf.nn.relu),
    layers.MaxPool2D(pool_size = [2, 2], strides = 2, padding = 'same'),
    
    layers.Conv2D(128, kernel_size = [3, 3], padding = 'same', activation = tf.nn.relu),
    layers.Conv2D(128, kernel_size = [3, 3], padding = 'same', activation = tf.nn.relu),
    layers.MaxPool2D(pool_size = [2, 2], strides = 2, padding = 'same'),
    
    layers.Conv2D(256, kernel_size = [3, 3], padding = 'same', activation = tf.nn.relu),
    layers.Conv2D(256, kernel_size = [3, 3], padding = 'same', activation = tf.nn.relu),
    layers.MaxPool2D(pool_size = [2, 2], strides = 2, padding = 'same'),
    
    layers.Conv2D(512, kernel_size = [3, 3], padding = 'same', activation = tf.nn.relu),
    layers.Conv2D(512, kernel_size = [3, 3], padding = 'same', activation = tf.nn.relu),
    layers.MaxPool2D(pool_size = [2, 2], strides = 2, padding = 'same'),
    
    layers.Conv2D(512, kernel_size = [3, 3], padding = 'same', activation = tf.nn.relu),
    layers.Conv2D(512, kernel_size = [3, 3], padding = 'same', activation = tf.nn.relu),
    layers.MaxPool2D(pool_size = [2, 2], strides = 2, padding = 'same'),
  
    '''
    After the convulutional units, the Full Connection Layers are build in VGG13.
    The last Dense layer has 100 output because the CIFAR100 datasets requires the machine to classify the objects into 100 different kinds.
    '''
    layers.Dense(256,activation = tf.nn.relu),
    layers.Dense(128,activation = tf.nn.relu),
    layers.Dense(100,activation = None),
])
