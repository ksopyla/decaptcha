# Convolutional neural network for captcha recognizing 

The project goal is to recognize captcha images with use of conv net. For network creation and trainning Tensorflow library was used.

Project is a playground for testing different architectures, training techniques, activation functions and optimizers.

## Data
Each image has 57x300 dim, but while loading it is padded with 0 to 64x304. It is better size for further processing by conv net.

The image label is simply file name.


### Trainning labes

Captchas' words has differnet lengts, starting from 3 chars up to 20. In order to have fixed vector size the labels were encoded as fixed size vectors. 
We assume that label contains max 20 characters, each character could be
'0...9A...Za...z_'
char 0 has an index 0
char A has an index 10 etc.
The vector contains only 0 and 1, each char position in word is encoded by 63 
continous vector positions, index 0 encodes occurence of char '0' at firs postion,
index 63 encodes occurence of char '0' at second position in the word

eg. word = 'at' looks like

vec[36]=1
vec[118]=1

because, at first position we have char 'a' so
vec[0:9]=0 it is reserved for digits
vec[10:35] is reserved for big latin letters
vec[36:61] is reserved for small latin letters, firs in alphabet is 'a' so it is at 36 position
vec[62] - reserved for '_'


character at second place 't'
vec[63:72] - digits
vec[73:98] - big letters
vec[99:124] - small letters, 't' is 19 letter in alphabet so 99+19=118
vec[125] - '_'


## Network architecure

The network is 6 layer Convolutional network:
* 3x3 conv - 1-> 32 (filter depth)
* 3x3 conv - 32-> 32
* max pool 
* dropout
* 3x3 conv - 32-> 64 (filter depth)
* 3x3 conv - 64-> 64
* max pool 
* dropout
* 3x3 conv - 64-> 64 (filter depth)
* max pool 
* dropout
* fully connected 1024 
* output - 20*63 (20 positions times 63 different chars )



