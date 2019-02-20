## Image Super Resolution using Generative Adversarial Networks

### Purpose:

![purpose](/img/purpose.jpg)

* Modifying based on ***[SRGAN](https://arxiv.org/abs/1609.04802)***

### *Generator*

#### 1. Feature extration part:

* ***Multiple kernel*** size and ***fusion*** ( for multiple ***receptive field*** and ***data character*** )

#### 2. Reconstruction part:

* Using ***upsizing convolution*** replace with pixelShuffle

* Adjust the number of layer and kernel size

#### 3. Loss funtion:

* Using ***SSIM*** as a loss function replace with MSE 

* Importing ***pretrained VGG19 first layer*** as a loss function

<br />

### *Discriminator*

* Decreasing number of layer

* Importing ***WGAN-GP*** strategy

<br />

### Architecture:

<br />

![Architecture](/img/architecture.png)

<br />

### Loss flow:

<br />

![LossFlow](/img/lossflow.png)

<br />

### Requirement:

```
python ver. 3.6.5
tensorflow ver. 1.8.0
tensorlayer ver. 1.8.5
```

<br />

### Data:

Samlpe:

![datasample](/img/sample.png)

<br />

* Collecting from STL10, source from ***[Stanford University CS](https://cs.stanford.edu/~acoates/stl10/)***
* training data:* ***[STL10 dataset](http://oomusou.io)***

