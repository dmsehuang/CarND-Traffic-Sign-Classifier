**Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./write_up_imgs/train_data.png "train data"
[image2]: ./write_up_imgs/valid_data.png "validation data"
[image3]: ./write_up_imgs/test_data.png "test data"
[image4]: ./write_up_imgs/image_processing.png "image processing"
[image5]: ./write_up_imgs/before_preprocess.png "before preprocess"
[image6]: ./write_up_imgs/after_preprocess.png "after preprocess"
[image7]: ./write_up_imgs/epoches.png "epoches"
[image8]: ./write_up_imgs/batch_size.png "batch size"
[image9]: ./write_up_imgs/dropout.png "dropout"
[image10]: ./test_images/11.jpg "11"
[image11]: ./test_images/21.jpg "21"
[image12]: ./test_images/24.jpg "24"
[image13]: ./test_images/27.jpg "27"
[image14]: ./test_images/28.jpg "28"

You're reading it! and here is a link to my [project code](https://github.com/dmsehuang/CarND-Traffic-Sign-Classifier/blob/traffic-sign-classifier/writeup.md)

### 1. Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set.

I used the pandas library to read the signnames.csv and used numpy to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799 
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the training, validation and test data distributes: 

![train data][image1]
![valid data][image2]
![test data][image3]

### Design and Test a Model Architecture

#### 1. Preprocessing pipeline.

The preprocessing pipeline contains the following steps:

1. Convert RGB image to grayscale image. The reason is that a grayscale image should contains enough information about the traffic sign.
2. Apply equalization to a grayscale image. The histogram of an image will distributed normally after equalization.
3. Normalize the grayscale image. The purpose of normalization is to make each feature has zero-mean and unit-variance.

Here is an example of an original, grayscale, equalized traffic sign image:

![image processing steps][image4]

Here is a statistic summary before and after preprocessing step:

![before preprocess][image5]
![after preprocess][image6]


#### 2. Architecture. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 5x5    	| 1x1 stride, valid padding, outputs 28x28x6 	|
| Dropout               | keep probability = 0.6                        |
| RELU					|												|
| Max pooling 2x2      	| 2x2 stride, same padding, outputs 14x14x6   	|
| Convolution 5x5	    | 1x1 stride, valid padding, output 10x10x16    |
| Dropout               | keep probability = 0.6                        |
| RELU					|												|
| Max pooling 2x2      	| 2x2 stride, same padding, outputs 5x5x16   	|
| Flatten               |                                               |
| Fully connected		| input = 1024, output = 120                    |
| Dropout               | keep probability = 0.6                        |
| Fully connected		| input = 120, output = 84                      |
| Dropout               | keep probability = 0.6                        |
| Fully connected		| input = 84, output = 43                       |
 


#### 3. Train the model

In order to train the model, I build couple tensors:
1. cross entropy. This tensor calculates the difference between a predicted logits and a label. It uses Softmax function to calculate the probability.
2. loss. This tensor calculates the total loss.
3. optimizer. AdamOptimizer is used to minize the loss. [Compare different optimizers](https://stackoverflow.com/questions/36162180/gradient-descent-vs-adagrad-vs-momentum-in-tensorflow)

The parameters I used to train the model is as follows:
* EPOCHS = 15
* BATCH_SIZE = 32 
* learning_rate = 0.001
* train_keep_prob = 0.6

#### 4. Enhance the model and tune the parameters 

My final model results were:
* training set accuracy of 98.5%
* validation set accuracy of 95%
* test set accuracy of 92.7%

Here are some exploration I've done:
1. Architecture. LeNet performs pretty well in classifying traffic signs. The only problem is that it may overfitting the trainning data. Thus, I added 4 dropout layers to the existing model.
2. Parameters. In order to tune the parameters, I basically plot the training and validation accuracy for a set of different parameters and find the best of it.

Here are some graphes that illustrate how I choose the parameters:

#### 1. EPOCHES

The training and validation accuracy tends to converge when the EPOCH = 15 since the curve is very flat at that position.
 
![epoches][image7]

#### 2. BATCH SIZE

The smaller the batch size it is, the higher the accuracy it will be. Even though the difference becomes smaller as epoch number becomes bigger.

![batches][image8] 

#### 3. Dropout

It seems that the dropout rate doesn't affect the validation accuracy very much. However, it's evident that with drop out rate of 40% (keep_prob = 0.6), the accuracy difference between training data and validation data is less and this means that the overfitting effect is smaller.

![dropout][image9]


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web.

Here are five German traffic signs that I found on the web:

![Right-of-way at the next intersection][image10] 
![Double curve][image11] 
![Road narrows on the right][image12] 
![Pedestrians][image13] 
![Children crossing][image14]

The second image is the most difficult one to classify since it looks very similar to "Dangerous curve to the left".

The third image is the most difficult one to classify since it's similar to "road narrows on the left" and "road narrows on both sides".

The rest of them should be easy enough to classify since they are of high quality.

#### 2. Discuss the prediction. 

Here are the results of the prediction:

| Image			                        | Prediction	        					| 
|:-------------------------------------:|:-----------------------------------------:| 
|Right-of-way at the next intersection  |Right-of-way at the next intersection      |
|Double curve                           |Double curve                               |
|Road narrows on the right              |Road narrows on the right                  |
|Pedestrians                            |Pedestrians                                |
|Children crossing                      |Children crossing                          |


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 92.7%. 

#### 3. How certain the model is. 

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

| Probability       | Prediction	        					| 
|:-----------------:|:-----------------------------------------:| 
|0.99               |Right-of-way at the next intersection      |
|0.21               |Double curve                               |
|0.64               |Road narrows on the right                  |
|0.57               |Pedestrians                                |
|0.97               |Children crossing                          |


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Apparently, from the feature map, we can see that the neural network tends to find the edges of a traffic signs and the contents within the sign.

