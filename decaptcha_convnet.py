'''
A Convolutional Network implementation example using TensorFlow library.
Author: ksopyla (krzysztofsopyla@gmail.com)
'''

import tensorflow as tf
import datetime as dt

import vec_mappings as vecmp


X, Y, captcha_text = vecmp.load_dataset()

# Parameters
learning_rate = 0.001
training_iters = 128*5000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 64*304 # captcha images has 64x304 size
n_classes = 20*63 # each word is encoded by 1260 vector
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Create model
def conv2d(img, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='SAME'),b))

def max_pool(img, k):
    return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def conv_net(_X, _weights, _biases, _dropout):
    # Reshape input picture
    _X = tf.reshape(_X, shape=[-1, 64, 304, 1])

    # Convolution Layer 5x5x32 first, layer with relu
    conv1 = conv2d(_X, _weights['wc1'], _biases['bc1'])
    
    # Convolution Layer 3x3x32, second layer with relu
    #conv1 = conv2d(conv1, _weights['wc11'], _biases['bc11'])

    # Max Pooling (down-sampling), change input size by factor of 2 
    conv1 = max_pool(conv1, k=2)
    # Apply Dropout
    conv1 = tf.nn.dropout(conv1, _dropout)
    
    
    # Convolution Layer, 5x5x64
    conv2 = conv2d(conv1, _weights['wc2'], _biases['bc2'])
    
    
    # Convolution Layer, 3x3x64
    #conv2 = conv2d(conv2, _weights['wc22'], _biases['bc22'])
    
    # Max Pooling (down-sampling)
    conv2 = max_pool(conv2, k=2)
    # Apply Dropout
    conv2 = tf.nn.dropout(conv2, _dropout)

    # Convolution Layer, 5x5x64
    conv3 = conv2d(conv2, _weights['wc3'], _biases['bc3'])
    
    # Max Pooling (down-sampling)
    conv3 = max_pool(conv3, k=2)
    # Apply Dropout
    conv3 = tf.nn.dropout(conv3, _dropout)

    # Fully connected layer
    dense1 = tf.reshape(conv3, [-1, _weights['wd1'].get_shape().as_list()[0]]) # Reshape conv2 output to fit dense layer input
    dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, _weights['wd1']), _biases['bd1'])) # Relu activation
    dense1 = tf.nn.dropout(dense1, _dropout) # Apply Dropout

    # Output, class prediction
    out = tf.add(tf.matmul(dense1, _weights['out']), _biases['out'])
    #out = tf.nn.softmax(out)
    return out

# Store layers weight & bias
weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])), # 5x5 conv, 1 input, 32 outputs
    #'wc11': tf.Variable(tf.random_normal([3, 3, 32, 32])), # 3x3 conv, 32 input, 32 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])), # 5x5 conv, 32 inputs, 64 outputs
    #'wc22': tf.Variable(tf.random_normal([3, 3, 64, 64])), # 5x5 conv, 32 inputs, 64 outputs
    'wc3': tf.Variable(tf.random_normal([3, 3, 64, 64])), # 3x3 conv, 64 inputs, 64 outputs
    'wd1': tf.Variable(tf.random_normal([8*38*64, 1024])), # fully connected, 64/(2*2*2)=8, 304/(2*2*2)=38 (three max pool k=2) inputs, 1024 outputs
    'out': tf.Variable(tf.random_normal([1024, n_classes])) # 1024 inputs, 10 outputs (class prediction)
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    #'bc11': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    #'bc22': tf.Variable(tf.random_normal([64])),
    'bc3': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer

????
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()


losses = list()
accuracies = list()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    
    epoch=0
    start_epoch=dt.datetime.now()
    
    
    
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_xs, batch_ys = vecmp.random_batch(X, Y, batch_size)
        # Fit training using batch data
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            accuracies.append(acc)
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            losses.append(loss)
            print "Iter " + str(step*batch_size) + " started={}".format(dt.datetime.now()) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc)
            

            epoch+=1
        
        step += 1
        
    end_epoch = dt.datetime.now()
    print "Optimization Finished, end={} duration={}".format(end_epoch,end_epoch-start_epoch)
    # Calculate accuracy for 256 mnist test images
    #print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.})
    

import matplotlib.pyplot as plt

# plt.plot(losses)
# plt.plot(accuracies)

plt.figure(1)
plt.subplot(211)
plt.plot(losses, '-b', label='Loss')
plt.title('Loss function')
plt.subplot(212)
plt.plot(accuracies, '-r', label='Acc')
plt.title('Accuracy')
    
