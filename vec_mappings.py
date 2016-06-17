import numpy as np



def load_dataset(folder="./shared/Captcha/img/"):
    
    #folder="./shared/Captcha/img/"
    
    file_list = os.listdir(folder)
    
    # number of files
    N =len([name for name in os.listdir(folder)])
    
    # stores images #N x 19456 (64*304)
    X = np.zeros([N, 64*304])
    
    # max captcha text = 20chars, at each postion coulb be 0...9A..Za..z_ so 63 different chars
    # 20x63= 1260
    Y = np.zeros([N, 20*63])
    
    
    for i, file in enumerate(file_list):
        
        path=folder+file
        img = imread(path)
        
        im_shape = img.shape
        
        if len(im_shape)>2:
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
        Y[i,:] = vecmap.map_words(captcha_text)
    return (X,Y)
    
    

def map_chars(c):
    """
    returns index/position of char passed in argument for sequence
    '0...9A...Za...z_'
    char 0 has an index 0
    char A has an index 10 etc.
    """
    
    if c =='_':
        k=62
        return k
    
    # compute index for digits '0'...'9', '0' has 0 index, '9' has 9
    # asci code for '0' is 48, so when we substract 48 from the char code we should
    # compute proper index for digits 0...9
    k= ord(c)-48
    
    if k>9:
        # char is not digit
        
        # we are computing index for 'A'...'Z'
        # code for char 'A' is 65, 'Z' is 90, and computing indexes for big letters
        # starts from 10 to 35
        # so we have to substract 65-10=55
        k=ord(c)-55
        if  k>35:
            # char is not big letter
            
            # we are computing index for 'a'...'z'
            # code for char 'a' is 97, 'z' is 122, and computing indexes for small letters
            # starts from 36 to 61
            # so we have to substract 97-36=61
            k=ord(c)-61
            if k >61:
                raise ValueError('wrong character') 
    
    return k    
    
    
def map_words(words):
    vector = np.zeros(20*63)
    
    for i, c in enumerate(words):
        idx=i*63+map_chars(c)
        
        vector[idx]=1
      
    return vector  
    
    
    