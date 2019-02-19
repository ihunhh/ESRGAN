## Image Super Resolution using Generative Adversarial Networks

### Base on SRGAN

### Modify:

### *Generator*

<br />

#### 1. Feature extration part:

* ***Multiple kernel*** size and ***fusion*** ( for multiple ***receptive field*** and ***data character*** )

<br />

#### 2. Reconstruction part:

* Using ***upsizing convolution*** replace with pixelShuffle

* Adjust the number of layer and kernel size

<br />

#### 3. Loss funtion:

* Using ***SSIM*** as a loss function replace with MSE 

* Using ***pretrained VGG19 first layer*** as a loss function

<br />

### *Discriminator*

<br />

* Decreasing number of layer

* import ***WGAN-GP*** strategy

<br />

___


### Architecture:

<br />

<br />

<br />

![Architecture](/img/ESRGAN2m.png)

___

### Loss flow:

<br />

<br />

<br />

![LossFlow](/img/lossflowm.png)

___

### Requirement:

<br />

+ python ver. 3.6.5
+ tensorflow ver. 1.8.0
+ tensorlayer ver. 1.8.5

<br />

### Data:

<br />

STL10, source from Stanford University
    
[STL](http://oomusou.io)

