## Traffic Sign Recognition Program
---

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./markup_images/train_set_histogram.png "Train Set Histogram"
[image2]: ./markup_images/train_set_sample.png "Train Set Sample"
[image3]: ./markup_images/before_and_after_normalize.png  "before and after normalizing"
[image4]: ./markup_images/new_images "New Traffic Signs"
[image5]: ./markup_images/lenet.png "LeNet Arch"
[image6]: ./markup_images/lenetmod.png "Modified LeNet Arch"
[image7]: ./markup_images/new_images_classifed.png "New Traffic Signs Classified"

[image8]: ./markup_images/speed_limit_50.png
[image9]: ./markup_images/speed_limit_60.png
[image10]: ./markup_images/yeild.png
[image11]: ./markup_images/slippery.png
[image12]: ./markup_images/turn_left_ahead.png



## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/messam2/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

Here is an exploratory visualization of the data set. It is a bar chart showing how the number of each class in the traning set.
Histogram for the training data set
![alt text][image1]

And this is a sample of the training set.
Random sample of the traning data set
![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to keep the three chanels of the image in order not to loos information from the image, only normlalize the data by subtracting the mean and divide by it.

Here is an example of a traffic sign image before and after normalizing.

![alt text][image3]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

 
Here is ana image for the final model.

![alt text][image6]

I used the same architecture in [this paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) and added drop out after first conventional layer and the second flat layer.


My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| (0) Input         		| 32x32x3 RGB image   							| 
| (1) Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| (2) RELU					|												|
| (3)Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| (4) dropout					|										0.5		|
| (5) Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16     									|
| (6) RELU					|												|
| (7) Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| (8) Convolution 5x5	    | 1x1 stride, valid padding, outputs 1x1x400     									|
| (9) RELU					|												|
| (10) Flatten 7 and 9					|									400, 400			|
| (11) Concatenate					|							800					|
| (12) dropout					|										0.5		|
| (13) Fully connected		| outputs 43 classes        									|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer.
hyperparameters:
EPOCHS=40
BATCH_SIZE=100
learning_rate=0.001
drop_out=0.9
mu=0
sigma=0.1

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the fifteenth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 100 %
* validation set accuracy of 93.6%
* test set accuracy of 93.6 %

First I used LeNet architecture from the last lab as a start point by modfying the model to take RGB image abd output 43 classes. However it didn't work well as it gave low validation acurrecy resulted from overfitting over the training data. So I chose sermanet model from that [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) however I added some modifications in the archticture such as:
* Adding dropout layer after the first convolution layer and another one after the last flatten layer to decrease the overfitting effect.
* Then from that instance I started to tune the parameters to get to the requried accuracy. The parameter that is tunned is the batch number 40, batch size I used 100 and dropout factor of 0.9.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image8] ![alt text][image9] ![alt text][image10]
![alt text][image11] ![alt text][image12]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit 50      		| speed limit 30   									| 
| Speed limit 60     			| Slippery road										|
| Yield					| Yield											|
| Slippery road	      		| Slippery road					 				|
| Turn left ahead			| Turn left ahead      							|

The model was able to correctly guess 24 of the 28 traffic signs, which gives an accuracy of 85.7%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 17th cell of the Ipython notebook.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
