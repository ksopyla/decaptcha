'''
A Convolutional Network implementation example using TensorFlow library.
Author: ksopyla (krzysztofsopyla@gmail.com)
'''

import tensorflow as tf
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import vec_mappings as vecmp
import optparse

def prepare_data(img_folder):
    
    X, Y, captcha_text = vecmp.load_dataset(folder=img_folder)

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

    return (X,Y,test_X,test_Y)


def save_plots(losses, train_acc, test_acc, training_iters,step,plot_title):
        
    # iters_steps
    iter_steps = [step *
                k for k in range((training_iters // step) + 1)]

    imh = plt.figure(1, figsize=(15, 12), dpi=160)
    # imh.tight_layout()
    # imh.subplots_adjust(top=0.88)

    imh.suptitle(plot_title)
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

    plt.savefig(plot_title)


def conv2d(img, w, b,acitivation_func='relu'):
    '''
    Creates 2d convolution layer with activation and bias
    img - tensor
    w - weights
    b - bias
    '''    

    if acitivation_func=='elu':
        return tf.nn.elu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='SAME'), b))
    else:
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='SAME'), b))
    


def max_pool(img, k):
    return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def model_2x2con_1con_1FC_weights(img_w, img_h,scale_weights=0.01):
    '''
    Create weights and do an initialization

    img_h - input image height
    img_w - input image width
    '''
        # Store layers weight & bias

    # relu initialization
    init_wc1 = np.sqrt(2.0 / (img_w * img_h)) # ~0.01
    init_wc11 = np.sqrt(2.0 / (3 * 3 * 32)) # ~0.08
    init_wc2 = np.sqrt(2.0 / (3 * 3 * 32)) # ~0.08
    init_wc21 = np.sqrt(2.0 / (3 * 3 * 64)) # ~0.06
    init_wc3 = np.sqrt(2.0 / (3 * 3 * 64))
    init_wd1 = np.sqrt(2.0 / (8 * 38 * 64)) #~0.01
    init_out = np.sqrt(2.0 / 1024) #~0.044

    #scale_weights = 'sqrt_HE'
    #scale_weights = 0.005
    init_wc1 = scale_weights
    init_wc11 = scale_weights
    init_wc2 = scale_weights
    init_wc21 = scale_weights
    init_wc3 = scale_weights
    init_wd1 = scale_weights
    init_out = scale_weights


    weights = {
        # 3x3 conv, 1 input, 32 outputs
        'wc1': tf.Variable(init_wc1 * tf.random_normal([3, 3, 1, 32])),
        # 3x3 conv, 32 input, 32 outputs
        'wc11': tf.Variable(init_wc11*tf.random_normal([3, 3, 32, 32])),
        # 3x3 conv, 32 inputs, 64 outputs
        'wc2': tf.Variable(init_wc2 * tf.random_normal([3, 3, 32, 64])),
        # 3x3 conv, 32 inputs, 64 outputs
        'wc21': tf.Variable(init_wc21*tf.random_normal([3, 3, 64, 64])),
        # 3x3 conv, 64 inputs, 64 outputs
        'wc3': tf.Variable(init_wc3 * tf.random_normal([3, 3, 64, 64])),
        # fully connected, 64/(2*2*2)=8, 304/(2*2*2)=38 (three max pool k=2)
        # inputs, 1024 outputs
        'wd1': tf.Variable(init_wd1 * tf.random_normal([8 * 38 * 64, 1024])),
        # 1024 inputs, 20*63 outputs for one catpcha word (max 20chars)
        'out': tf.Variable(init_out * tf.random_normal([1024, 20 * 63]))
    }

    bias_scale = 0.01
    biases = {
        'bc1':  tf.Variable(bias_scale * tf.random_normal([32])),
        'bc11': tf.Variable(bias_scale * tf.random_normal([32])),
        'bc2':  tf.Variable(bias_scale * tf.random_normal([64])),
        'bc21': tf.Variable(bias_scale * tf.random_normal([64])),
        'bc3':  tf.Variable(bias_scale * tf.random_normal([64])),
        'bd1':  tf.Variable(bias_scale * tf.random_normal([1024])),
        'out':  tf.Variable(bias_scale * tf.random_normal([20 * 63]))
    }

    return weights, biases

def model_2x2con_1con_1FC(_X, _dropout, img_h, img_w, scale_weights=0.01):
    """
    Creates tensorflow net model, adds layers

    X - tensor data
    _weights - initailized weights

    img_h - input image height
    img_w - input image width
    
    """
    _weights, _biases = model_2x2con_1con_1FC_weights(img_w, img_h,scale_weights)
    
    # Reshape input picture
    _X = tf.reshape(_X, shape=[-1, img_h, img_w, 1])

    # Convolution Layer 3x3x32 first, layer with relu
    conv1 = conv2d(_X, _weights['wc1'], _biases['bc1'])
    # Convolution Layer 3x3x32, second layer with relu
    conv1 = conv2d(conv1, _weights['wc11'], _biases['bc11'])
    # Max Pooling (down-sampling), change input size by factor of 2
    conv1 = max_pool(conv1, k=2)
    # Apply Dropout
    conv1 = tf.nn.dropout(conv1, _dropout)

    # Convolution Layer, 3x3x64
    conv2 = conv2d(conv1, _weights['wc2'], _biases['bc2'])
    # Convolution Layer, 3x3x64
    conv2 = conv2d(conv2, _weights['wc21'], _biases['bc21'])
    # Max Pooling (down-sampling)
    conv2 = max_pool(conv2, k=2)
    # Apply Dropout
    conv2 = tf.nn.dropout(conv2, _dropout)

    # Convolution Layer, 3x3x64
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

def model_3x3con_1FC(_X, _dropout, img_h, img_w, scale_weights=0.1):
    """
    Simpler conv model with 3x2 conv (3x3) layers

    X - tensor data
    _weights - initailized weights

    img_h - input image height
    img_w - input image width
    
    """
    #scale_weights = 'sqrt_HE'
    init_wc1 = scale_weights
    init_wc11 = scale_weights
    init_wc2 = scale_weights
    init_wc21 = scale_weights
    init_wc3 = scale_weights
    init_wd1 = scale_weights
    init_wfc1 = scale_weights
    init_out= scale_weights

    bias_scale = 0.1
    
    # 3x3 conv, 1 input, 64 outputs
    wc1 = tf.Variable(init_wc1 * tf.random_normal([3, 3, 1, 64]))
    bc1 = tf.Variable(bias_scale * tf.random_normal([64]))
    # 3x3 conv, 64 input, 64 outputs
    wc11 = tf.Variable(init_wc11*tf.random_normal([3, 3, 64, 64]))
    bc11 = tf.Variable(bias_scale * tf.random_normal([64]))
    
    # 3x3 conv, 64 inputs, 96 outputs
    wc2 = tf.Variable(init_wc2 * tf.random_normal([3, 3, 64, 96]))
    bc2 = tf.Variable(bias_scale * tf.random_normal([96]))
    # 3x3 conv, 96 inputs, 96 outputs
    wc21 = tf.Variable(init_wc21*tf.random_normal([3, 3, 96, 96]))
    bc21 = tf.Variable(bias_scale * tf.random_normal([96]))
    
    # 3x3 conv, 64 inputs, 64 outputs
    wc3 = tf.Variable(init_wc3 * tf.random_normal([3, 3, 96, 128]))
    bc3 = tf.Variable(bias_scale * tf.random_normal([128]))
    wc31 = tf.Variable(init_wc3 * tf.random_normal([3, 3, 128, 128]))
    bc31 = tf.Variable(bias_scale * tf.random_normal([128]))
    
    # fully connected, 64/(2*2*2)=8, 304/(2*2*2)=38 (three max pool k=2)
    # inputs, 2048 outputs
    out_size=512
    wfc1 = tf.Variable(init_wfc1 * tf.random_normal([8 * 38 * 128, out_size]))
    bf1 = tf.Variable(bias_scale * tf.random_normal([out_size]))

    # 1024 inputs, 20*63 outputs for one catpcha word (max 20chars)
    wout =  tf.Variable(init_out * tf.random_normal([out_size, 20 * 63]))
    bout = tf.Variable(bias_scale * tf.random_normal([20 * 63]))

    
    # Reshape input picture
    _X = tf.reshape(_X, shape=[-1, img_h, img_w, 1])

    # Convolution Layer 3x3
    conv1 = conv2d(_X, wc1, bc1)
    conv1 = conv2d(conv1, wc11, bc11)
    # Max Pooling (down-sampling), change input size by factor of 2
    conv1 = max_pool(conv1, k=2)
    
    # Convolution Layer
    conv2 = conv2d(conv1, wc2, bc2)
    conv2 = conv2d(conv2, wc21, bc21)
    # Max Pooling (down-sampling)
    conv2 = max_pool(conv2, k=2)
    
    # Convolution Layer,
    conv3 = conv2d(conv2, wc3, bc3)
    conv3 = conv2d(conv3, wc31, bc31)
    # Max Pooling (down-sampling)
    conv3 = max_pool(conv3, k=2)
    
    # Fully connected layer
    # Reshape conv2 output to fit dense layer input
    fc1 = tf.reshape(conv3, [-1, wfc1.get_shape().as_list()[0]])
    # Relu activation
    fc1 = tf.nn.relu(tf.matmul(fc1, wd1)+ bfc1)
    fc1 = tf.nn.dropout(fc1, _dropout)  # Apply Dropout
    
    # Output, class prediction
    out = tf.matmul(fc1, wout)+bout
    #out = tf.nn.softmax(out)
    return out




def main(learning_r=0.001, drop=0.7,train_iters=20000,):

    print('Learning script with params learning_rate={}, dropout={}, iterations={}'.format(learning_r,drop,train_iters))
    
    #img_folder = '/home/ksopyla/dev/data/data_07_2016/'
    #img_folder = '/home/ksirg/dev/data/data_07_2016/'
    img_folder = './shared/Captcha/data_07_2016/img/'
    ds_name = 'data_07_2016'
    
    X,Y,test_X, test_Y = prepare_data(img_folder)
    test_size = test_X.shape[0]

    # Parameters
    learning_rate = learning_r
    dropout = drop  # Dropout, probability to keep units
    training_iters = train_iters  # 15000 is ok
    batch_size = 64

    display_step = 100

    # Network Parameters
    img_h = 64
    img_w = 304
    n_input = img_h * img_w  # captcha images has 64x304 size
    n_classes = 20 * 63  # each word is encoded by 1260 vector


    # tf Graph input
    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

    # Construct model 1 - not so good, convergence problems
    # scale_weights=0.005
    # pred = model_2x2con_1con_1FC(x, keep_prob,img_h,img_w,scale_weights)
    # model_version = model_2x2con_1con_1FC.__name__
    

    # Construct model 2 - much simpler with 3x2 conv layers, dropout only at fc layer
    scale_weights=0.1
    pred = model_3x3con_1FC(x, keep_prob,img_h,img_w,scale_weights)
    model_version = model_3x3con_1FC.__name__


    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(pred, y)
    loss = tf.reduce_mean(cross_entropy)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    opt_alg = 'adam'

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
                batch_trainX, batch_trainY, idx = vecmp.random_batch(X, Y,100)
                
                trn_acc = sess.run(accuracy, feed_dict={
                            x: batch_trainX, y: batch_trainY, keep_prob: 1.})
                train_acc.append(trn_acc)

                #print("loss start {}".format(dt.datetime.now()))
                # Calculate batch loss
                batch_loss = sess.run(
                    loss, feed_dict={x: batch_trainX, y: batch_trainY, keep_prob: 1.})
                losses.append(batch_loss)
                
                # Calculate accuracy on random test batch 
                batch_testX, batch_testY, idx = vecmp.random_batch(test_X, test_Y, 100)
                
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
                #save_file = './models/model_{}_init_{}.ckpt'.format(ds_name,scale_weights)
                #save_path = saver.save(sess, save_file)

        end_epoch = dt.datetime.now()
        print("Optimization Finished, end={} duration={}".format(
            end_epoch, end_epoch - start_epoch))

        # Calculate accuracy
        print("\n\nStart testing...")
        parts = 10
        test_batch_sz= test_size//parts
        i=0
        k=0
        acc=0.0
        for part in range(parts):
            i = k
            k = i+test_batch_sz
            batch_test_X= test_X[i:k]
            batch_test_Y = test_Y[i:k]
            batch_acc= sess.run(accuracy, feed_dict={x: batch_test_X, y: batch_test_Y, keep_prob: 1.})
            acc+=batch_acc

            print("Batch #{} accuracy= {}, predictions:".format(part,batch_acc))
            pp = sess.run(pred, feed_dict={x: batch_test_X, y: batch_test_Y, keep_prob: 1.})
            p = tf.reshape(pp, [-1, 20, 63])
            max_idx_p = tf.argmax(p, 2).eval()
            l = tf.reshape(test_Y, [-1, 20, 63])
            # max idx acros the rows
            max_idx_l = tf.argmax(l, 2).eval()

            for k in range(test_batch_sz):

                true_word = vecmp.map_vec_pos2words(max_idx_l[k, :])
                predicted_word = vecmp.map_vec_pos2words(max_idx_p[k, :])

                got_error = ''
                if(true_word != predicted_word):
                    got_error = '<--- error'
                print("true : {}, predicted {} {}".format(
                    true_word, predicted_word, got_error))
        
        acc= acc/parts
        print("Testing Accuracy:{}".format(acc))


        plot_title = './plots/captcha_{}_{}_opt_{}_lr_{}_dropout_{}_6l_init_{}_iter_{}.png'.format(ds_name,model_version,opt_alg,learning_rate,dropout,scale_weights,training_iters)
        save_plots(losses, train_acc, test_acc, training_iters,display_step, plot_title)





if __name__ == "__main__":
    # set command line options
    
    print('in main')
    import sys
    print(sys.argv)
    
    parser = optparse.OptionParser('usage: %prog [options]')
    parser.add_option('-d', '--dropout',
                      dest='dropout',
                      default=0.7,
                      type='float',                      
                      help='dropout')
    parser.add_option('-l', '--learning_rate',
                      dest='learning_rate',
                      default=0.001,
                      type='float',
                      help='optimizer learning rate')
    parser.add_option('-i', '--training_iters',
                      dest='training_iters',
                      default=20000,
                      type='int',
                      help='number of training iteration')

    parser.add_option('-a', '--activation_func',
                      dest='activation_func',
                      default='relu',
                      help='relu, elu')
   
    parser.add_option('-f', '--ipython_kernel_log',
                      dest='iptyhon_kernel', help='ipython kernel log')                        
                      



    (options, args) = parser.parse_args()
    learning_rate = options.learning_rate
    dropout = options.dropout
    training_iters = options.training_iters
    main(learning_r=learning_rate,drop=dropout,train_iters=training_iters)