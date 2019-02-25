## Image Super Resolution using Generative Adversarial Networks

### Purpose:

![purpose](/img/purpose.jpg)

* **Modifying based on** ***[SRGAN](https://arxiv.org/abs/1609.04802)***
* **upsizing image to** ***4x***

<br />

### Architecture:

<br />

![Architecture](/img/architecture.png)

<br />

### *Generator*

#### 1. Feature extration part:

* ***Multiple kernel*** size and ***fusion*** ( for multiple ***receptive field*** and ***data character*** )

* 1x1 convolution for ***concentrating feature***

#### 2. Reconstruction part:

* Using ***upsizing convolution*** replace with pixelShuffle

* Adjust the number of layer and ***kernel size***

#### 3. Loss funtion:

* Using ***SSIM*** as a loss function replace with MSE 

* Importing ***pretrained VGG19 first layer*** as a loss function

<br />

### *Discriminator*

* **Decreasing** number of layer

* Importing ***WGAN-GP*** strategy

<br />

### Loss flow:

<br />

![LossFlow](/img/lossflow.png)

<br />

### Requirement:

```
python ver. 3.6.5
tensorflow ver. 0.12.0-rc1
```

<br />

### Data:

* Samlpe:

   ![datasample](/img/sample.png)

* My input size is ***12x12***, ground truth is ***48x48***, resize from STL10(original size is 96x96)

* Collecting from STL10, source from ***[Stanford University](https://cs.stanford.edu/~acoates/stl10/)***

    * ***[training dataset](https://drive.google.com/file/d/1FQxb7fFC2A-taChBujBf9-4cfpbNfUty/view?usp=sharing)***
  
    * ***[testing dataset](https://drive.google.com/file/d/1T2nCA9sTozLz1Rarc7kRfiABXKMMdCHv/view?usp=sharing)***
    
<br />

### Pre-trained model:

* ***myWork:***

   * Putting them in ***checkpoint*** folder

      * ***10000 epoch***
   
         * ***[generator](https://drive.google.com/file/d/1wHzn3cu6U1tVEkkQ2VoqQnjeCA4ZyDHN/view?usp=sharing)***
      
         * ***[discriminator](https://drive.google.com/file/d/1mWa5jJIYcvIsgWcjcOaKkszwrR51FIsF/view?usp=sharing)***
         
* ***VGG19:***
   
   * ***[pre-trained model](https://drive.google.com/file/d/1p4KTvPjGrrGqB78poKst9CA4CavHtPfu/view?usp=sharing)***
   

   
  
<br />
  
### Procedure:
  
* Putting training and testing data in ***data2017***
  
   * ***training***
  
      ```python main.py```
    
   * ***testing***
  
      ```python main.py --mode=testing```
      
 <br />
  
### Results:

* ***Image:***

   <img src="/img/results_img.jpg" height="400">


* ***Comparison with other methods:***

   ![results_sheet](/img/results_sheet.jpg)
