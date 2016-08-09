
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import matplotlib.pyplot as plt

def getSize(txt, font):
    testImg = Image.new('RGB', (1, 1))
    testDraw = ImageDraw.Draw(testImg)
    return testDraw.textsize(txt, font)






fontname = "Generate/OpenSans-Regular.ttf"
fontsize = 24   
font = ImageFont.truetype(fontname, fontsize)
text = "Hello world"

colorText = "black"
colorOutline = "red"
colorBackground = "white"



#################################3

img = Image.new('RGB', (width+40, height+40), colorBackground)
d = ImageDraw.Draw(img)
d.text( (5, 5), text,  font=font, fill=colorText)
w=txt.rotate(17.5,  expand=1)

plt.imshow(w)

#########################

width, height = font.getsize(text)

image1 = Image.new('RGBA', (200, 150), colorBackground)
draw1 = ImageDraw.Draw(image1)
draw1.text((10, 10), text=text, font=font, fill=colorText)
plt.imshow(image1)

image2 = Image.new('RGBA', (width+10, height+10), colorBackground)
draw2 = ImageDraw.Draw(image2)
draw2.text((5, 0), text=text, font=font, fill=colorText)

plt.imshow(image2)

image2 = image2.rotate(30, expand=1)

px, py = 10, 10
sx, sy = image2.size
image1.paste(image2, (px, py, px + sx, py + sy), image2)

plt.imshow(image1)


#####

import numpy.random as rnd

numbers = [3,4,5,8]

img = Image.new('RGBA', (96, 32), colorBackground)

patch_size=32
dh=9 #height offset

digit_offset=0

for i,nr in enumerate(numbers):
    
    digit_str = str(nr)
    fw, fh=font.getsize(digit_str)
    im1 = Image.new('RGBA',(fw,fh),colorBackground)
    #im1 = Image.new('RGBA',(patch_size,patch_size),colorBackground)
    
    d1  = ImageDraw.Draw(im1)
    
    d1.text( (0,-dh),digit_str,font=font, fill=colorText)
    #d1.rectangle((0, 0, w, h), outline=colorOutline)
    
    angle = rnd.randint(-30,30)    
    #im1_rot=im1.rotate(angle, resample=Image.BILINEAR,  expand=1)
    im1_rot=im1.rotate(angle, resample=Image.BICUBIC,  expand=1)
    print 'im1_rot size', im1_rot.size

    pad_w = rnd.randint(1,6)
    pad_h = rnd.randint(5)
    
    pos_w = digit_offset+pad_w
    print pos_w
    img.paste(im1_rot,(pos_w,pad_h),im1_rot)
    
    
    digit_offset=pos_w+im1_rot.size[0]
    print digit_offset

plt.imshow(img)



