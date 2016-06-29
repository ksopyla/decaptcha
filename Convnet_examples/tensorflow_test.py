import tensorflow as tf



learning_rate = 0.001

#define a variable to hold normal random values 
normal_rv = tf.Variable( tf.truncated_normal([2,3],stddev = 0.1))

vec1 = tf.Variable([1,2,3,4,5])


vec2 = tf.Variable(np.array([ [ 0, 5, 1, 0, 1, 0, 0, 2, 1, 0], [ 2, 3, 0, 0, 1, 0, 1, 2, 1, 0], [ 0, 0, 4, 0, 1, 2, 0, 2, 1, 0] ] ))
labels = tf.Variable(np.array([ [ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [ 0, 0, 1, 0, 0, 0, 0, 0, 0, 0] ] ))


pred = tf.Variable(np.array([ [ 0, 9, 1, 6, 2, 2, 7, 2, 1, 0, 3, 7], 
                              [ 9, 1, 0, 2, 7, 1, 0, 8, 2, 0, 6, 4], 
                              [ 0, 9, 1, 6, 1, 3, 1, 8, 1, 0, 6, 4] ] ))
                              
lab = tf.Variable(np.array([  [ 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1], 
                              [ 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0], 
                              [ 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1] ] ))


pred.initializer.run()
lab.initializer.run()


n_classes=10
# weights = {
#     'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])), # 5x5 conv, 1 input, 32 outputs
#     'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])), # 5x5 conv, 32 inputs, 64 outputs
#     'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])), # fully connected, 7*7*64 inputs, 1024 outputs
#     'out': tf.Variable(tf.random_normal([1024, n_classes])) # 1024 inputs, 10 outputs (class prediction)
# }

#initialize the variable
init_op = tf.initialize_all_variables()

#run the graph
with tf.Session() as sess:
    sess.run(init_op) #execute init_op
    # #print the random values that we sample
    # print (sess.run(normal_rv))
    # wd= weights['wd1'].get_shape().as_list()
    
    # print(sess.run(vec1))
    # print(vec1.eval())
    
    # # [2,3]
    # print tf.slice(vec1,[1],[2]).eval()
    
    s = tf.split(0, 3, vec2)
    s0 = tf.to_float(s[0])
    
    rs = tf.reshape(s0, [1,8]).eval()
    print tf.shape(rs)
    print tf.nn.softmax(rs).eval()
    
    ed = tf.expand_dims(s0,1 )
    print tf.shape(ed)
    print tf.nn.softmax(ed).eval()
    
    #k= [tf.nn.softmax(tf.to_float(c)) for c in s]

    #print tf.softmax(s0)

    flab = tf.to_float(labels)
    tf.nn.softmax_cross_entropy_with_logits(tl,flab)
    
    
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
    
    
    
    p = tf.reshape(pred,[3,4,3])
    #max idx acros the rows
    max_idx=tf.argmax(p,2).eval()
    
    l = tf.reshape(lab,[3,4,3])
    #max idx acros the rows
    max_idx=tf.argmax(l,2).eval()
    
    
