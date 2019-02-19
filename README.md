## Image Super Resolution using Generative Adversarial Networks

### Base on SRGAN

### Modify:

#### 1. feature extration part:


#### 2. reconstruction part:


#### 3. loss funtion:

* import SSIM as a loss function replace with MSE 

* import WGAN

1. llll



### Architecture:

![Architecture](/img/ESRGAN2m.png)

=======

### Loss flow:

![LossFlow](/img/lossflowm.png)

### Requirement:

+ python ver. 3.6.5
+ tensorflow ver. 1.8.0
+ tensorlayer ver. 1.8.5


