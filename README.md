## Image Super Resolution using Generative Adversarial Networks

### Base on SRGAN

### Modify:

#### 1. feature extration part:

* multiple kernel size and concate

<font color=#0099ff size=7 face="黑体">color=#0099ff size=72 face="黑体"</font>

* 



#### 2. reconstruction part:

* using upsizing convolution replace with pixelShuffle

* adjust the number of layer and kernel size


#### 3. loss funtion:

* using SSIM as a loss function replace with MSE 

* using pretrained VGG19 layer one as a loss function

___


### Architecture:

![Architecture](/img/ESRGAN2m.png)

___

### Loss flow:

![LossFlow](/img/lossflowm.png)

___

### Requirement:

+ python ver. 3.6.5
+ tensorflow ver. 1.8.0
+ tensorlayer ver. 1.8.5


