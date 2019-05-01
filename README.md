## Image Super Resolution using Generative Adversarial Networks

### Purpose:

<img src="/img/purpose.jpg" width="666">

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

#### 2. Residual block:

* Using ***ELU***  as the activation function for better ***mapping ability***

* Increasing more convolutional layer

#### 3. Reconstruction part:

* Replace pixelShuffle with ***upsizing convolution*** for ***reducing*** computing overhead

* Adjust the number of layer and ***kernel size*** for better ***reconstruction*** performance

#### 4. Loss funtion:

* Using ***SSIM*** as a loss function to approach ***visual acceptance of human beings***

<br />

### *Discriminator*

* **Decreasing** number of layers

* Importing ***gradient penalty*** strategy to improve the ability
of ***discriminator***

<br />

### Loss flow:

<br />

<img src="/img/lossflow.png" width="800">

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

* My input size of images is ***12x12***, ground truth of images is ***48x48***, downsized from **STL10** dataset(original size is 96x96)

* Selecting from **STL10** by ***[Stanford University](https://cs.stanford.edu/~acoates/stl10/)***

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
  
* Putting training and testing data in ***data2017*** folder
  
   * ***training***
  
      ```python main.py```
    
   * ***testing***
  
      ```python main.py --mode=testing```
      
<br />
  
### Results:

* ***Image:***

   <img src="/img/results_img.jpg" width="800">


* ***Comparison with other methods:***

   ![results_sheet](/img/results_sheet.jpg)
   
<br />
 
### Future work:
   
   ![future work](/img/futurework.png)
   
   * ***Detail:***
   
      * ***Remove VGG loss*** to reduce the ***dependency*** or ***mutual exclusion*** between loss functions
      * Modifying the ***SSIM loss***, because if the loss function is a ***convex function***, that will help convergence  
      * Trying import ***octave convolution*** to tune the ***high and low frequency*** signal ratio for better generator
      
<br />

### Referrence:

[1]	Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang."Image Super-Resolution Using Deep Convolutional Networks" Image, Video, and Multidimensional Signal Processing Workshop (IVMSP)， 2015.

[2]	Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, Wenzhe Shi. “Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network” Computer Vision and Pattern Recognition (CVPR), 2017.

[3]	Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio. “Generative Adversarial Networks” 2014.

[4]	Karen Simonyan, Andrew Zisserman. “Very Deep Convolutional Networks for Large-Scale Image	Recognition”  Pattern Recognition (ACPR), 2015.

[5]	Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun “Deep Residual Learning for Image Recognition” 	Computer Vision and Pattern Recognition, 2015.

[6]	Martin Arjovsky, Soumith Chintala, Léon Bottou. “Wasserstein GAN” arXiv.org Machine Learning (stat.ML); Machine Learning (cs.LG) 2017.

[7]	Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron Courville ” Improved Training of Wasserstein GANs ” arXiv.org Machine Learning (cs.LG); Machine Learning (stat.ML) 2017.

[8]	Zhou Wang, Member, Alan Conrad Bovik, Fellow, Hamid Rahim Sheikh, Student Member, and Eero P. Simoncelli. ”Image Quality Assessment: From Error Visibility to Structural Similarity” IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600-612, Apr. 2004.

[9]	Zhou Wang, Eero P. Simoncelli and Alan C. Bovik. “Multi-scale Structural Similarity for Image Quality Assessment” The Thrity-Seventh Asilomar Conference on Signals, Systems & Computers, 2003.

[10]	Adam Coates, Honglak Lee, Andrew Y. Ng An Analysis of Single Layer Networks in Unsupervised Feature Learning AISTATS, 2011.
