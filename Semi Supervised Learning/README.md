
# Semi-Supervised Learning

Deep supervised learning has been key to a lot of success in computer vision research over the past decade. However,  its dependence on carefully labelled data, which is both expensive and time consuming to acquire, limits its potential for a lot of practical applications.

Semi-supervised learning describes a class of algorithms that seek to learn from both unlabeled and labeled samples, typically assumed to be sampled from the same or similar distributions. 

In taking a semi-supervised approach, we can train a classifier on the small amount of labeled data, and then use the classifier to make predictions on the unlabeled data. Since these predictions are likely better than random guessing, the unlabeled data predictions can be adopted as ‘pseudo-labels’ in subsequent iterations of the classifier. While there are many flavors of semi-supervised learning, this specific technique is called self-training.

# Approach to implement Semi-Supervised Learning for image classification 
Step 1: Split the labeled data instances into train and test sets. Then, train a deep learning model on the labeled training data.

Step 2: Use the trained DL model to predict class labels for all of the unlabeled data instances. Of these predicted class labels, the ones with the highest probability of             being correct are adopted as ‘pseudo-labels’.

Step 3: Concatenate the ‘pseudo-labeled’ data with the labeled training data. Re-train the model on the combined ‘pseudo-labeled’ and labeled training data.

Step 4: Use the trained classifier to predict class labels for the labeled test data instances. Evaluate classifier performance using your metric(s) of choice.


# About the dataset used
I have used the [STL-10 dataset](https://cs.stanford.edu/~acoates/stl10/) for this task. 

The STL-10 dataset is an image recognition dataset for developing unsupervised feature learning, deep learning, self-taught learning algorithms. It is inspired by the [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html) dataset but with some modifications. In particular, each class has fewer labeled training examples than in CIFAR-10, but a very large set of unlabeled examples is provided to learn image models prior to supervised training. The primary challenge is to make use of the unlabeled data (which comes from a similar but different distribution from the labeled data) to build a useful prior.

#### Dataset overview
* This dataset contains 10 classes: airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck.
* Images are 96x96 pixels, color.
* 500 training images (10 pre-defined folds), 800 test images per class.
* 100000 unlabeled images for unsupervised learning. These examples are extracted from a similar but broader distribution of images. For instance, it contains other types of     animals (bears, rabbits, etc.) and vehicles (trains, buses, etc.) in addition to the ones in the labeled set.
* Images were acquired from labeled examples on ImageNet.

# Training model (Supervised Learning)
We will have to train a model in a supervised manner first before using the model to predict class labels for unlabeled images. I first instantiated a VGG-19 model with pre-trained weights, froze it (so that its weights are not updated during training) and added 1 fully connected dense layer on top of it along with a dropout layer and softmax layer to predict the label(s).

Trained the model on Kaggle’s TPU
Results: accuracy on test set was 66.56%

#  Using pseudo-labeling
1. Predicted the label of   a batch of unlabelled images using the model trained on labelled data and then appended the image and its respective label to the training dataset      if the probability of the label being correct was greater than 0.99.
2. Pseudo-labelled the unlabelled images in batches of 100.
3. Trained the model again on this augmented training set. Repeated this 4 times for a single batch. (Did not use the entire unlabelled images dataset to restrict training        time, only used around 6.5k/100k unlabelled images for pseudolabelling.)

Results: accuracy on test set after pseudo-labelling was 70.71%

Link to notebook - https://www.kaggle.com/vedantdalimkar/semi-supervised-learning-final 
(Notebook code does not have comments)

Link to final saved .h5 model (after pseudolabelling) - https://drive.google.com/drive/folders/1F9rKzHF4fq4i0VTfr4-91HUgYJF9AFJR?usp=sharing
