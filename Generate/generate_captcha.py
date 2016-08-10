
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import matplotlib.pyplot as plt

import time


def getSize(txt, font):
    testImg = Image.new('RGB', (1, 1))
    testDraw = ImageDraw.Draw(testImg)
    return testDraw.textsize(txt, font)


def genDigitsImg(numbers,font,img_size=(64,32), colorBackground = "white",    colorText = "black"):
    '''
    Generates one image with random digits with specified font
    
    numbers - numpy array with digits
    img_size - tuple of img width and height
    
    font - PIL font object
    
    Returns
    ===========
    img - PIL img object
    
    '''
    digit_offset=5
    dh=9 #height offset
    angle_var=20


    img = Image.new('RGBA', img_size, colorBackground)

    for i,nr in enumerate(numbers):
        
        digit_str = str(nr)
        fw, fh=font.getsize(digit_str)
        im1 = Image.new('RGBA',(fw,fh),colorBackground)
        #im1 = Image.new('RGBA',(patch_size,patch_size),colorBackground)
        
        d1  = ImageDraw.Draw(im1)
        
        d1.text( (0,-dh),digit_str,font=font, fill=colorText)
        #d1.rectangle((0, 0, w, h), outline=colorOutline)
        
        angle = rnd.randint(-angle_var,angle_var)    
        #im1_rot=im1.rotate(angle, resample=Image.BILINEAR,  expand=1)
        im1_rot=im1.rotate(angle, resample=Image.BICUBIC,  expand=1)
        #print 'im1_rot size', im1_rot.size
    
        pad_w = rnd.randint(-5,6)
        pad_h = rnd.randint(10)
        
        pos_w = digit_offset+pad_w
        #print pos_w
        img.paste(im1_rot,(pos_w,pad_h),im1_rot)
        
        
        digit_offset=pos_w+im1_rot.size[0]
        #print digit_offset
        
    return img
    
    
    
    



fontname = "Generate/OpenSans-Regular.ttf"
fontsize = 26   
font = ImageFont.truetype(fontname, fontsize)


colorText = "black"
colorBackground = "white"

#####

import numpy.random as rnd
patch_size=32
dh=9 #height offset
angle_var=20

folder='shared/Digits_2/'


for a in range(20):

    numbers = rnd.choice(10,2, replace=True)    
    numbers_str = ''.join([str(x) for x in numbers])
    digit_offset=5
    
    img = genDigitsImg(numbers,font,img_size=(56,32))
    
    # img = Image.new('RGBA', (56, 32), colorBackground)

    # for i,nr in enumerate(numbers):
        
    #     digit_str = str(nr)
    #     fw, fh=font.getsize(digit_str)
    #     im1 = Image.new('RGBA',(fw,fh),colorBackground)
    #     #im1 = Image.new('RGBA',(patch_size,patch_size),colorBackground)
        
    #     d1  = ImageDraw.Draw(im1)
        
    #     d1.text( (0,-dh),digit_str,font=font, fill=colorText)
    #     #d1.rectangle((0, 0, w, h), outline=colorOutline)
        
    #     angle = rnd.randint(-angle_var,angle_var)    
    #     #im1_rot=im1.rotate(angle, resample=Image.BILINEAR,  expand=1)
    #     im1_rot=im1.rotate(angle, resample=Image.BICUBIC,  expand=1)
    #     #print 'im1_rot size', im1_rot.size
    
    #     pad_w = rnd.randint(-5,6)
    #     pad_h = rnd.randint(10)
        
    #     pos_w = digit_offset+pad_w
    #     #print pos_w
    #     img.paste(im1_rot,(pos_w,pad_h),im1_rot)
        
        
    #     digit_offset=pos_w+im1_rot.size[0]
    #     #print digit_offset
    
    #img.save('{}{}_{}.png'.format(folder,numbers_str,int(time.time())))
    plt.imshow(img)
    plt.show()
    time.sleep(0.5)



