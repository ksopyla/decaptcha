import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
import os
import vec_mappings as vecmap


folder="/home/ksopyla/dev/data/captcha_img/"

#probability of showing the images, higher shows less images
show_prob=0.99

file_list = os.listdir(folder)

#stores all capchas img sizes (57x300)
img_sizes = list()

#stores all img letter numbers
letters=list()

max_letters=0
avg_letters=0.0
counter=0.0


# number of files
N =len([name for name in os.listdir(folder)])

# stores images #N x 19456 (64*304)
X = np.zeros([N, 64*304])

# max captcha text = 20chars, at each postion coulb be 0...9A..Za..z_ so 63 different chars
# 20x63= 1260
Y = np.zeros([N, 20*63])


for i, file in enumerate(file_list):
    
    cur_letters = len(file)
    letters.append(cur_letters)
    counter+=1
    max_letters = max(max_letters, cur_letters)
    avg_letters +=cur_letters
    path=folder+file
    img = imread(path)
    
    im_shape = img.shape
    
    if len(im_shape)>2:
        print(file)
        #break
    
        #convert to gray img
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        
        #convert ot gray, faster
        img = np.mean(img,-1)

    
    else: 
        # each immage has size 57x300, we have to 
        # resize images to 64x304, because conv nets work better with 
        # dimensions divided by 2, we add 4 pixesl at the top
        # 3 pixesl at the bottom and 2 to left and right
        # TODO: change it to more automatic way, what if image size will be 
        # different than 57.300?
        im_pad = np.pad(img,((4,3),(2,2)), 'constant', constant_values=(255,))
        
    
    # file with padded text with '_', each captach has 20 char lenght
    captcha_text = file.ljust(20,'_')
    
    X[i,:] = im_pad.flatten()
    Y[i,:] = vecmap.map_words2vec(captcha_text)
    
    
    
    
    if(np.random.random()> show_prob):
        #plt.imshow(img,cmap='Greys_r')
        plt.imshow(im_pad,cmap='Greys_r')
        #plt.imshow(im_pad,cmap=plt.cm.binary)
        plt.show()
        
#        time.sleep(1)
        
        
    else:
        #break
        pass
    
avg_letters/=counter
# np.histogram(img_sizes)
# plt.hist(img_sizes)

np.histogram(letters, bins=range(0,21))
plt.hist(letters, bins=range(0,25))


