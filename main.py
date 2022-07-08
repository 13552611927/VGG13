import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential

def main():
    #building the network
    conv_net = Sequential(conv_layer)
    
    fc_net = Sequential([
        layers.Dense(256,activation = tf.nn.relu),
        layers.Dense(128,activation = tf.nn.relu),
        layers.Dense(100,activation = None),
    ])
    
    
    conv_net.build(input_shape = [None,32,32,3])
    fc_net.build(input_shape = [None,512])
    optimizer = optimizers.Adam(lr = 1e-4)
    
    variables =  conv_net.trainable_variables + fc_net.trainable_variables
    
    #importing the training datasets
    for epoch in range(50):
        for step, (x,y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                out = conv_net(x)
                out = tf.reshape(out, [-1, 512])
                logits = fc_net(out)
                y_onehot = tf.one_hot(y, depth = 100)
                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits = True)
                loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(grads, variables))
            if step %100 == 0:
                print(epoch, step, 'loss:', float(loss))
        
        
        total_num = 0
        total_correct = 0
        for x,y in test_db:
            out = conv_net(x)
            out = tf.reshape(out,[-1,512])
            logits = fc_net(out)
            prob = tf.nn.softmax(logits, axis=1)
            preb = tf.argmax(prob,axis=1)
            preb = tf.cast(pred,dtype = tf.int32)
            
            correct = tf.cast(tf.equal(preb, y),dtype=tf.int32)
            correct = tf.reduce_sum(correct)
            
            total_num += x.shape[0]
            total_correct += int(correct)
            
        acc = total_correct / total_num
        print(epoch, 'accuracy:', acc)
