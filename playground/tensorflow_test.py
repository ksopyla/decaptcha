import tensorflow as tf
import numpy as np



learning_rate = 0.001

#define a variable to hold normal random values 
normal_rv = tf.Variable( tf.truncated_normal([2,3],stddev = 0.1))

vec1 = tf.Variable([1,2,3,4,5])


vec2 = tf.Variable(np.array([ [ 0, 5, 1, 0, 1, 0, 0, 2, 1, 0], [ 2, 3, 0, 0, 1, 0, 1, 2, 1, 0], [ 0, 0, 4, 0, 1, 2, 0, 2, 1, 0] ] ))
labels = tf.Variable(np.array([ [ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [ 0, 0, 1, 0, 0, 0, 0, 0, 0, 0] ] ))


#size of pred batchx(3*4), in each row we have 4 softmax groups (by 3)
pred = tf.Variable(np.array([ [ 0, 9, 1, 6, 2, 2, 7, 2, 1, 0, 3, 7], 
                              [ 9, 1, 0, 2, 7, 1, 0, 8, 2, 0, 6, 4], 
                              [ 0, 9, 1, 6, 1, 3, 1, 8, 1, 0, 6, 4] ] ))
                              
# size = batch x (3*4), we have 3 batch rows, each row contains 4 outputs with 3 classes                              
lab = tf.Variable(np.array([  [ 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1], 
                              [ 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0], 
                              [ 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1] ] ))



sess = tf.InteractiveSession()
pred.initializer.run()
lab.initializer.run()

pred = tf.to_float(pred)
lab = tf.to_float(lab)

spred = tf.split(1,4,pred)
slab = tf.split(1,4,lab)

costs = list()
for i in range(4):
    costs.append(tf.nn.softmax_cross_entropy_with_logits(spred[i],slab[i]))
    
    
rcosts = list()
for i in range(4):
    rcosts.append(tf.reduce_mean(costs[i]))
    
loss = tf.reduce_sum(rcosts)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

sess.run(optimizer)


#batch, rows, cols
p = tf.reshape(pred,[3,4,3])
#max idx acros the rows
max_idx_p=tf.argmax(p,2).eval()

l = tf.reshape(lab,[3,4,3])
#max idx acros the rows
max_idx_l=tf.argmax(l,2).eval()

correct_pred = tf.equal(max_idx_p,max_idx_l)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    
#close session
sess.close()