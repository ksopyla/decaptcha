'''
A Convolutional Network implementation example using TensorFlow library.
Author: ksopyla (krzysztofsopyla@gmail.com)
'''

import tensorflow as tf
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import vec_mappings as vecmp

#img_folder = '/home/ksopyla/dev/captcha_data/data_07_2016/'
img_folder = './shared/Captcha/data_07_2016/img/'

X, Y, captcha_text = vecmp.load_dataset(folder=img_folder)

ds_name = 'data_07_2016'

# invert and normalize to [0,1]
#X =  (255- Xdata)/255.0

# standarization
# compute mean across the rows, sum elements from each column and divide
x_mean = X.mean(axis=0)
x_std = X.std(axis=0)
X = (X - x_mean) / (x_std + 0.00001)

test_size = min(1000, X.shape[0])
random_idx = np.random.choice(X.shape[0], test_size, replace=False)

test_X = X[random_idx, :]
test_Y = Y[random_idx, :]

X = np.delete(X, random_idx, axis=0)
Y = np.delete(Y, random_idx, axis=0)


# Parameters
learning_rate = 0.001
batch_size = 64
training_iters = 35000  # 15000 is ok
display_step = 1000

# Network Parameters
img_h = 64
img_w = 304
n_input = img_h * img_w  # captcha images has 64x304 size
n_classes = 20 * 63  # each word is encoded by 1260 vector
dropout = 0.75  # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

# Create model


def conv2d(img, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='SAME'), b))


def max_pool(img, k):
    return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def conv_net(_X, _weights, _biases, _dropout):
    # Reshape input picture
    _X = tf.reshape(_X, shape=[-1, img_h, img_w, 1])

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
    # Reshape conv2 output to fit dense layer input
    dense1 = tf.reshape(conv3, [-1, _weights['wd1'].get_shape().as_list()[0]])
    # Relu activation
    dense1 = tf.nn.relu(
        tf.add(tf.matmul(dense1, _weights['wd1']), _biases['bd1']))
    dense1 = tf.nn.dropout(dense1, _dropout)  # Apply Dropout

    # Output, class prediction
    out = tf.add(tf.matmul(dense1, _weights['out']), _biases['out'])
    #out = tf.nn.softmax(out)
    return out

# Store layers weight & bias

# relu initialization
init_wc1 = np.sqrt(2.0 / (img_w * img_h))
init_wc2 = np.sqrt(2.0 / (3 * 3 * 32))
init_wc3 = np.sqrt(2.0 / (3 * 3 * 64))
init_wd1 = np.sqrt(2.0 / (8 * 38 * 64))
init_out = np.sqrt(2.0 / 1024)

alpha = 'sqrt_HE'
#init_wc1 = alpha
#init_wc2 = alpha
#init_wc3 = alpha
#init_wd1 = alpha
#init_out =  alpha


weights = {
    # 3x3 conv, 1 input, 32 outputs
    'wc1': tf.Variable(init_wc1 * tf.random_normal([3, 3, 1, 32])),
    #'wc11': tf.Variable(alpha*tf.random_normal([3, 3, 32, 32])), # 3x3 conv, 32 input, 32 outputs
    # 3x3 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(init_wc2 * tf.random_normal([3, 3, 32, 64])),
    #'wc22': tf.Variable(alpha*tf.random_normal([3, 3, 64, 64])), # 3x3 conv, 32 inputs, 64 outputs
    # 3x3 conv, 64 inputs, 64 outputs
    'wc3': tf.Variable(init_wc3 * tf.random_normal([3, 3, 64, 64])),
    # fully connected, 64/(2*2*2)=8, 304/(2*2*2)=38 (three max pool k=2)
    # inputs, 1024 outputs
    'wd1': tf.Variable(init_wd1 * tf.random_normal([8 * 38 * 64, 1024])),
    #'out': tf.Variable(alpha*tf.random_normal([1024, n_classes]))
    # 1024 inputs, 20*63 outputs for one catpcha word (max 20chars)
    'out': tf.Variable(init_out * tf.random_normal([1024, 20 * 63]))
}

biases = {
    'bc1': tf.Variable(0.1 * tf.random_normal([32])),
    #'bc11': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(0.1 * tf.random_normal([64])),
    #'bc22': tf.Variable(tf.random_normal([64])),
    'bc3': tf.Variable(0.1 * tf.random_normal([64])),
    'bd1': tf.Variable(0.1 * tf.random_normal([1024])),
    'out': tf.Variable(0.1 * tf.random_normal([20 * 63]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer


############
# splited softmax_cross_entropy loss
# split prediction for each char it takes 63 continous postions, we have 20 chars
# split_pred = tf.split(1,20,pred)
# split_y = tf.split(1,20,y)


# #compute partial softmax cost, for each char
# costs = list()
# for i in range(20):
#     costs.append(tf.nn.softmax_cross_entropy_with_logits(split_pred[i],split_y[i]))

# #reduce cost for each char
# rcosts = list()
# for i in range(20):
#     rcosts.append(tf.reduce_mean(costs[i]))

# # global reduce
# loss = tf.reduce_sum(rcosts)


cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(pred, y)
loss = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


# Evaluate model

# pred are in format batch_size,20*63, reshape it in order to have each character prediction
# in row, then take argmax of each row (across columns) then check if it is equal
# original label max indexes
# then sum all good results and compute mean (accuracy)

#batch, rows, cols
p = tf.reshape(pred, [-1, 20, 63])
# max idx acros the rows
# max_idx_p=tf.argmax(p,2).eval()
max_idx_p = tf.argmax(p, 2)

l = tf.reshape(y, [-1, 20, 63])
# max idx acros the rows
# max_idx_l=tf.argmax(l,2).eval()
max_idx_l = tf.argmax(l, 2)

correct_pred = tf.equal(max_idx_p, max_idx_l)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
#correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

losses = list()
train_acc = list()
test_acc = list()

saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 0
    epoch = 0
    start_epoch = dt.datetime.now()

    # Keep training until reach max iterations
    while step <= training_iters:
        batch_xs, batch_ys, idx = vecmp.random_batch(X, Y, batch_size)

        # Fit training using batch data

        start_op = dt.datetime.now()

        sess.run(optimizer, feed_dict={
                 x: batch_xs, y: batch_ys, keep_prob: dropout})
        end_op = dt.datetime.now()
        #print("#{} opt step {} {} takes {}".format(step,start_op,end_op, end_op-start_op))

        if step % display_step == 0:

            #print("acc start {}".format(dt.datetime.now()))
            # Calculate accuracy on random training samples
            batch_trainX, batch_trainY, idx = vecmp.random_batch(X, Y, 500)
            
            trn_acc = sess.run(accuracy, feed_dict={
                           x: batch_trainX, y: batch_trainY, keep_prob: 1.})
            train_acc.append(trn_acc)

            #print("loss start {}".format(dt.datetime.now()))
            # Calculate batch loss
            batch_loss = sess.run(
                loss, feed_dict={x: batch_trainX, y: batch_trainY, keep_prob: 1.})
            losses.append(batch_loss)
            
            # Calculate accuracy on random test batch 
            batch_testX, batch_testY, idx = vecmp.random_batch(test_X, test_Y, 500)
            
            tst_acc = sess.run(accuracy, feed_dict={
                           x: batch_testX, y: batch_testY, keep_prob: 1.})
            test_acc.append(tst_acc)

            print("##Iter {}, Minibatch Loss={}, Train Acc={} Test Acc={}".format(step, batch_loss,trn_acc,tst_acc))

            batch_idx = 0
            k = idx[batch_idx]

            pp = sess.run(pred, feed_dict={
                          x: batch_trainX, y: batch_trainY, keep_prob: 1.})
            p = tf.reshape(pp, [-1, 20, 63])
            max_idx_p = tf.argmax(p, 2).eval()

            predicted_word = vecmp.map_vec_pos2words(max_idx_p[batch_idx, :])

            l = tf.reshape(batch_trainY, [-1, 20, 63])
            # max idx acros the rows
            max_idx_l = tf.argmax(l, 2).eval()
            true_word = vecmp.map_vec_pos2words(max_idx_l[batch_idx, :])

            print("true : {}, predicted {}".format(true_word, predicted_word))

            epoch += 1

        step += 1

        if step % 5000 == 0:
            print('saving...')
            save_file = './models/model_{}_init_{}.ckpt'.format(ds_name,alpha)
            save_path = saver.save(sess, save_file)

    end_epoch = dt.datetime.now()
    print("Optimization Finished, end={} duration={}".format(
        end_epoch, end_epoch - start_epoch))

    # Calculate accuracy
    print("Testing Accuracy:{}".format(
        sess.run(accuracy, feed_dict={x: test_X, y: test_Y, keep_prob: 1.})))

    pp = sess.run(pred, feed_dict={x: test_X, y: test_Y, keep_prob: 1.})
    p = tf.reshape(pp, [-1, 20, 63])
    max_idx_p = tf.argmax(p, 2).eval()
    l = tf.reshape(test_Y, [-1, 20, 63])
    # max idx acros the rows
    max_idx_l = tf.argmax(l, 2).eval()

    for k in range(test_size):

        true_word = vecmp.map_vec_pos2words(max_idx_l[k, :])

        predicted_word = vecmp.map_vec_pos2words(max_idx_p[k, :])

        got_error = ''
        if(true_word != predicted_word):
            got_error = '<--- error'
        print("true : {}, predicted {} {}".format(
            true_word, predicted_word, got_error))




# iters_steps
iter_steps = [display_step *
              k for k in range((training_iters / display_step) + 1)]

trainning_version = './plots/captcha_{}_acc_4l_init_{}_iter_{}.png'.format(ds_name,alpha,training_iters)


imh = plt.figure(1, figsize=(15, 12), dpi=160)
# imh.tight_layout()
# imh.subplots_adjust(top=0.88)

imh.suptitle(trainning_version)
plt.subplot(311)
#plt.plot(iter_steps,losses, '-g', label='Loss')
plt.semilogy(iter_steps, losses, '-g', label='Loss')
plt.title('Loss function')
plt.subplot(312)
plt.plot(iter_steps, train_acc, '-r', label='Trn Acc')
plt.title('Train Accuracy')

plt.subplot(313)
plt.plot(iter_steps, test_acc, '-r', label='Tst Acc')
plt.title('Test Accuracy')


plt.tight_layout()
plt.subplots_adjust(top=0.88)

plt.savefig(trainning_version)
