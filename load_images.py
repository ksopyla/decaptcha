from scipy import ndimage
from PIL import Image          
import time


folder="./shared/Captcha/img/"
file = "1864_shste"
path= folder+file
img = Image.open(path)
img.show() 


import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
import os
import pandas as pd


#probability of showing the images, higher shows less images
show_prob=0.9


file_list = os.listdir(folder)
df=pd.DataFrame({ 'file_name':  file_list})

#stores all capchas img sizes (57x300)
img_sizes = list()

#stores all img letter numbers
letters=list()

max_letters=0
avg_letters=0.0
counter=0.0


import time

for file in file_list:
    
    cur_letters = len(file)
    letters.append(cur_letters)
    counter+=1
    max_letters = max(max_letters, cur_letters)
    avg_letters +=cur_letters
    path=folder+file
    img = imread(path)
    
    im_shape = img.shape
    
    if len(im_shape)>2:
        print file
        break
    
        #convert to gray img
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        
        #convert ot gray, faster
        img = np.mean(img,-1)

    
    else: 
        #resize image to 64.304, because conv nets work better with 
        # dimensions divided by 2, previous dims=(57,300), we add 4 pixesl at the top
        # 3 pixesl at the bottom and 2 to left and right
        im_pad = np.pad(img,((4,3),(2,2)), 'constant', constant_values=(255,))
        
    
    
    if(np.random.random()> show_prob):
        plt.imshow(img,cmap='Greys_r')
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


