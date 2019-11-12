## Machine Learning Algorithms

![img](https://scotch-res.cloudinary.com/image/upload/w_1050,q_auto:good,f_auto/v1545808446/yo7vwyhuuwm7m8p4cjjy.png)


# 1. Supervised Learning

## 1.1. Classification

In classification problems we split input examples by certain characteristic.

Usage examples: spam-filters, language detection, finding similar documents, handwritten letters recognition, etc.

### 1.1.1. Decision Trees

[example code](https://github.com/vuurball/machine-learning/blob/master/Algorithms/SupervisedLearning/Classification/DecisionTrees.md)

### 1.1.2. SVM (support vector machine) 

[example code](https://github.com/vuurball/machine-learning/blob/master/Algorithms/SupervisedLearning/Classification/SVM.py)

### 1.1.3. K-NN 

[example code](https://github.com/vuurball/machine-learning/blob/master/Algorithms/SupervisedLearning/Classification/K-NN.py)

For classifying an object into 1 of the pre known catagories.
The algo will look at the K nearest objects, and based on those will assign the category of the majority of the neighbor objects.
it is important to select the right K. for 2 catagories, K shouldnt be =2, because then there might be a deadlock, same as for 3 categories, K shouldnt be =3.
the KNN model will not only output a classification, but it will also give an accuracy % of the model and an confidence % for each classification.
so if we had 2 categories A,B, and K=3, if the model tries to classify X, and finds nearest 3 neighbors are A,A,B. then classification is A with confidence of 66%.
This is not the most afficient algo because each classification will recalculated distances of all points in data set.
SVM is much more scalable.


## 1.2. Regression

In regression problems we do real value predictions. Basically we try to draw a line/plane/n-dimensional plane along the training examples.

Usage examples: stock price forecast, sales analysis, dependency of any number, etc.


### 1.2.1. Linear Regression 

[example code](https://github.com/vuurball/machine-learning/blob/master/Algorithms/SupervisedLearning/Regression/LinearRegression.md)

### 1.2.2. Logistic Regression

[example code](https://github.com/vuurball/machine-learning/blob/master/Algorithms/SupervisedLearning/Regression/LogisticRegression.md)

To predict whether a person will buy a car (1) or (0)
To know whether the tumor is malignant (1) or (0)


# 2. Unsupervised Learning

The Algo finds the clusters (labels) on its own without the scientist feeding them to the model first.

Usage examples: market segmentation, social networks analysis, organize computing clusters, astronomical data analysis, image compression, etc.

## 2.1. Clustering


### 2.1.1. K-Means

[Flat clusteting] providing the model with a dataset and asking it to sepparate the dataset into K number of groups

[example code](https://github.com/vuurball/machine-learning/blob/master/Algorithms/UnsupervisedLearning/Clustering/K-Means.py)

### 2.1.2. Mean Shift

[Hierarchical clustering] providing the model with a dataset and asking it to sepparate the dataset into groups, telling us how many groups there are, and what they are

[example code](https://github.com/vuurball/machine-learning/blob/master/Algorithms/UnsupervisedLearning/Clustering/MeanShift.py)


