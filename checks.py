# this file is unnecesery
# It containse some sanity checks to run in interactive environment like plon.io or ipython

import numpy as np
import matplotlib.pyplot as plt
import vec_mappings as vecmp


# load batch
batch_idx=0

batch_xs, batch_ys, idx = vecmp.random_batch(X, Y, batch_size)

batch_cap = [ captcha_text[i] for i in idx]

k=idx[batch_idx]

np.sum(X[k,:]!= batch_xs[batch_idx,:])
np.sum(Y[k,:]!= batch_ys[batch_idx,:])


#show text
captcha_text[k]
vecmp.map_vec2words(batch_ys[batch_idx,:])


#show images

for i in range(5):
    k=idx[i]
    title = batch_cap[i]
    plt.imshow(X[k,:].reshape(64,304),cmap='Greys_r')
    plt.title(title)
    plt.show()
    print('batch img')
    plt.imshow(batch_xs[i,:].reshape(64,304),cmap='Greys_r')
    plt.title(title)
    plt.show()

########################3
# run optimizer
sess = tf.InteractiveSession()
sess.run(init)
saver = tf.train.Saver()
import datetime as dt
for i in range(101):
    batch_xs, batch_ys, idx = vecmp.random_batch(X, Y, batch_size)
    # Fit training using batch data
    
    print("#{} opt step {}".format(i,dt.datetime.now()))
    sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
    print("end step {}\n".format(dt.datetime.now())) 
    
    if i%20 ==0:
        save_path = saver.save(sess, "./model.ckpt")
        
        
###################


#run network, you have to create session and run the pred network definition in decapcha_convnet.py
pp = sess.run(pred, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})


# reshape for extract batches and make row for each character
p = tf.reshape(pp,[batch_size,20,63])
max_idx_p=tf.argmax(p,2).eval()


l = tf.reshape(batch_ys,[batch_size,20,63])
#max idx acros the rows
max_idx_l=tf.argmax(l,2).eval()
#max_idx_l=tf.argmax(l,2)

correct_pred = tf.equal(max_idx_p,max_idx_l)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



###############################
# loss func
#split prediction for each char it takes 63 continous postions, we have 20 chars

split_pred = tf.split(1,20,pp)
split_y = tf.split(1,20,batch_ys)

sp1=split_y[0].eval()
for i in range(64):
    check_match = batch_cap[i][0]==vecmp.map_vec2words(sp1[i])
    print "{} true={} split={}".format(check_match,batch_cap[i][0],vecmp.map_vec2words(sp1[i]))



#tf.nn.sigmoid_cross_entropy_with_logits

#compute partial softmax cost, for each char
costs = list()
for i in range(20):
    costs.append(tf.nn.softmax_cross_entropy_with_logits(split_pred[i],split_y[i]))
    
#reduce cost for each char
rcosts = list()
for i in range(20):
    rcosts.append(tf.reduce_mean(costs[i]))
    
# global reduce    
loss = tf.reduce_sum(rcosts)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
