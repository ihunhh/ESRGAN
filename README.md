## Image Super Resolution using Generative Adversarial Networks

### Base on SRGAN

### Modify:

#### 1. feature extration part:

* multiple kernel size and concate

** iii

* 



#### 2. reconstruction part:

* using upsizing convolution replace with pixelShuffle

* adjust the number of layer and kernel size


#### 3. loss funtion:

* using SSIM as a loss function replace with MSE 

* using pretrained VGG19 layer one as a loss function

___________________


### Architecture:

![Architecture](/img/ESRGAN2m.png)

=======

### Loss flow:

![LossFlow](/img/lossflowm.png)

### Requirement:

+ python ver. 3.6.5
+ tensorflow ver. 1.8.0
+ tensorlayer ver. 1.8.5


