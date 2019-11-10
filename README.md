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

### 1.1.3. Logistic Regression 

### 1.1.4. Naive Bayes 

### 1.1.5. K-NN 

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

### 1.2.1. Polynomial Regression

### 1.2.2. Linear Regression 

[example code](https://github.com/vuurball/machine-learning/blob/master/Algorithms/SupervisedLearning/Regression/LinearRegression.md)

### 1.2.3. Ridge/Lasso Regression


# 2. Unsupervised Learning

The Algo finds the clusters (labels) on its own without the scientist feeding them to the model first.

Usage examples: market segmentation, social networks analysis, organize computing clusters, astronomical data analysis, image compression, etc.

## 2.1. Clustering

**Flat clusteting:** we tell the machine to find 2 or 3 clusters

**Hierarchical clustering:** the machine figures out what groups there are and how many there are

### 2.1.1. K-Means

[Flat clusteting] providing the model with a dataset and asking it to sepparate the dataset into K number of groups

[example code](https://github.com/vuurball/machine-learning/blob/master/Algorithms/UnsupervisedLearning/Clustering/K-Means.py)

### 2.1.2. Mean Shift

[Hierarchical clustering] providing the model with a dataset and asking it to sepparate the dataset into groups, telling us how many groups there are, and what they are

[example code](https://github.com/vuurball/machine-learning/blob/master/Algorithms/UnsupervisedLearning/Clustering/MeanShift.py)

### 2.1.3. Fuzzy C-Means

### 2.1.4. DBSCAN

### 2.1.5. Agglomerative

## 2.2. Association Rule Learning

### 2.2.1. FP Growth

### 2.2.2. Euclat

### 2.2.3. Apriori

## 2.3 Dimensionality Reduction

### 2.3.1. t-SNE

### 2.3.2. PCA

### 2.3.3. LSA

### 2.3.4. SVD

### 2.3.5. LDA

# 3. Reinforcement Learning


# 4. Neural Networks & Deep Learning

The neural network itself isn't an algorithm, but rather a framework for many different machine learning algorithms to work together and process complex data inputs.

Usage examples: as a substitute of all other algorithms in general, image recognition, voice recognition, image processing _(applying specific style_), language translation, etc.

# 5. Ensemble Learning

## 5.1. Stacking

## 5.2 Bagging

### 5.2.1 Random Forrest

## 5.3. Boosting

### 5.3.1. XGBoost

### 5.3.2. LightGBM

### 5.3.3. CatBoost

### 5.3.4. AdaBoost
