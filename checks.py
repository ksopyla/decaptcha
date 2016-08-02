# this file is unnecesery
# It containse some sanity checks to run in interactive environment like plon.io or ipython

import numpy as np
import matplotlib.pyplot as plt
import vec_mappings as vecmp


# load batch
batch_idx=0

batch_xs, batch_ys, idx = vecmp.random_batch(X, Y, batch_size)


k=idx[batch_idx]

np.sum(X[k,:]!= batch_xs[batch_idx,:])
np.sum(Y[k,:]!= batch_ys[batch_idx,:])


#show text
captcha_text[k]
vecmp.map_vec2words(batch_ys[batch_idx,:])


#show images
plt.imshow(X[k,:].reshape(64,304),cmap='Greys_r')
plt.imshow(batch_xs[batch_idx,:].reshape(64,304),cmap='Greys_r')


sess = tf.InteractiveSession()
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



# check how spliting is done
split_y2 = tf.split(1,20,batch_ys)
splits2=list()
for sp in split_y2:
    splits2.append(sp.eval())

nsplits2= np.array(splits2)    



chars_pos = max_idx_l[0,:]
word=list()

for i, c in enumerate(chars_pos):
    char_at_pos = i #c/63
    char_idx = c%63
    
    if char_idx<10:
        char_code= char_idx+ ord('0')
    elif char_idx <36:
        char_code= char_idx-10 + ord('A')
    elif char_idx < 62:
        char_code= char_idx-36 + ord('a')
    elif char_idx == 62:
        char_code = ord('_')
    else:
        raise ValueError('not recognized char code')
        
    
    word.append(chr(char_code))
  
"".join(word)
