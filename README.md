# Exam 2 for ISC_5935 Data Science meets Health Sciences, Spring 2023

The objective of this exam is to evaluate our current learning in the following topics:
*Image filters and resampling of medical image formats, basic concepts of supervised machine learning,*
*neural networks and basics of Convolutional Layers.*

## Question 1 (Image filters and resampling 5 Pts)
Use ITK to resample `data/cor.mha` into isotropic voxels of size 0.5 'spacing' units. 

What are the dimensions of the resampled image? 

## Question 2 (Supervised machine learning 10 Pts)
For each of the models  ([Model 1](Model_1.ipynb) and [Model 2](Model_2.ipynb) answer the following questions:

1. What is the model architecture?
2. What is the loss function?
3. Is the last model overfitting or underfitting? Why?
5. Does the last model have high bias or high variance? Why?
6. At which epoch would you stop training with training provided?
7. What suggestions would you give to improve each model (if any)?

## Question 3 (Neural networks and data exploration 10 Pts)
Considering the data from FILE `data/diabetes.csv` containing  the publicly available Pima Indians Diabetes 
dataset from the UCI Machine Learning Repository, design a neural network that predicts the onset of diabetes.

Answer the following questions about the dataset and your proposed solution:

1. Which information is available in the dataset?
2. How many patients are in the dataset?
3. What is the percentage of patients with diabetes?
4. What is the range of values for each feature (min and max)?
5. How you will preprocess the data?
6. How will you split the data into training and testing?
7. What architecture will you use for the neural network? (Provide information like the # of layers, # of neurons per layer, 
activation functions, etc. explain your choices)
8. What will be a suitable loss function?

## Question 4 (CNNs 10 Pts)
Assume you have the image [](imgs/cnn.png) as input and a 3x3 convolutional filter with all 1's in the
first row, 0's on the second row, and 2's on the third row. Please answer the following questions:

* If you have no padding and a stride of 1, what is the **size** of the output of applying a convolutional layer with this kernel?
* If you have no padding and a stride of 2, what is the **size** of the output of applying a convolutional layer with this kernel?
* If you have a padding of 1 and a stride of 1, what is the **size** of the output of applying a convolutional layer with this kernel?
* If we want 8 filters, padding of 'same', and stride and dilation of 1, in our CNN layer, what is the **size** of the output?

## Question 5 (Extra 10 Pts)
Implement the proposed architecture in Question 3 and train it. Evaluate the model and report the results.