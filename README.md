# Fundamentals of Deep Learning

Taught by [Terence Parr](https://explained.ai/) and [Yannet Interian](https://www.usfca.edu/faculty/yannet-interian).

This course teaches the fundamentals of deep learning, starting with a crash course in supervised learning and an overview of neural network architecture. After the "chalk talk" overview, the remainder of the course walks through a number of notebooks that you can use as templates to get started on your own applications. We use PyTorch for implementation examples.

**Prerequisites**. We assume a familiarity with linear and logistic regression and the basic mathematics of line fitting.  The target audience of this course is [MS data science students at the University of San Francisco](https://www.usfca.edu/arts-sciences/graduate-programs/data-science) after they have completed about one semester of work.

## Part One

Lectures by [Terence Parr](https://explained.ai/) with video comments by [Yannet Interian](https://www.usfca.edu/faculty/yannet-interian).

### Overview slides

At the start of the seminar, Terence goes through a crash course in machine learning and the basics of deep learning. See [Crash course slides (PDF)](lectures/crashcourse.pdf) or [Crash course slides (PPTX)](lectures/crashcourse.pptx).

The videos associated with the following notebooks are available here: [Youtube playlist](https://www.youtube.com/playlist?list=PLFCc_Fc116ikeol9CZcWWKqmrJljxhE4N). (The video and audio quality is lousy due to zoom compression, my editing, and recompression.) Here are some direct links associated with the PowerPoint slides:

* [Machine learning overview](https://youtu.be/rAw842OlkfA)
* [Deep learning regressors](https://youtu.be/gA97mSPQzDk)
* [Deep learning classifiers](https://youtu.be/qU5YfOa_LeI)
* [Training neural networks](https://youtu.be/SIO53mkB7Js)

### Notebooks

1. [intro-regression-training-cars.ipynb](notebooks/1.intro-regression-training-cars.ipynb)&nbsp;&nbsp;&nbsp;[Video](https://youtu.be/mUf3tLUOXy4)<br>Load toy cars data set and train regression models to predict miles per gallon (MPG) through a variety of techniques. We start out doing a brute force grid search of many different slope and intercept (m, b) model parameters, looking for the best fit. Then we manually compute partial derivatives of the loss function and perform gradient descent using plain numpy. We look at the effect on the loss function of normalizing numeric variables to have zero mean and standard deviation one. Finally, this notebook shows you how to use the autograd (auto differentiation) functionality of pytorch as a way to transition from numpy to pytorch training loops.
2. [pytorch-nn-training-cars.ipynb](notebooks/2.pytorch-nn-training-cars.ipynb)&nbsp;&nbsp;&nbsp;[Video](https://youtu.be/4F-EklhbzWM)<br>Once we can implement our own gradient descent using pytorch autograd and matrix algebra, it's time to graduate to using pytorch's built-in neural network module and the built-in optimizers (e.g., Adam). Next, we observe how a sequence of two linear models is effectively the same as a single linear model. After we add a nonlinearity, we see more sophisticated curve fitting. Then we see how a sequence of multiple linear units plus nonlinearities affects predictions. Finally, we see what happens if we give a model too much power: the regression curve over fits the training data.
3. [train-test-diabetes.ipynb](notebooks/3.train-test-diabetes.ipynb)&nbsp;&nbsp;&nbsp;[Video](https://youtu.be/7aW9IheaIUQ)<br>This notebook explores how to use a validation set to estimate how well a model generalizes from its training data to unknown test vectors. We will see that deep learning models often have so many parameters that we can drive training loss to zero, but unfortunately the validation loss grows as the model overfits. We will also compare how deep learning does compared to a random forest model as a baseline.
4. [batch-normalization.ipynb](notebooks/4.batch-normalization.ipynb) (*skip as I decided it's not the right time to introduce this concept*)<br>Just as we normalize or standardize the input variables, networks train better if we normalize the output of each layer's activation. It is called batch normalization because we normally train with batches of records not the entire data set, but it's really just fixing the mean and variance of each layer activation to zero mean and unit variance.
5. [binary-classifier-wine.ipynb](notebooks/5.binary-classifier-wine.ipynb)&nbsp;&nbsp;&nbsp;[Video](https://youtu.be/K1GnPoirl1s)<br>Shifting to binary classification now, we consider the toy wine data set and build models that use features proline and alcohol to predict wine classification (class 0 or class 1). We will add a sigmoid activation function to the final linear layer, which will give us the probability that an input vector represents class 1. A single linear layer plus the sigmoid yields a standard logistic regression model. By adding another linear layer and nonlinearity, we see a curved decision boundary between classes. By adding lots of neurons and more layers, we see even more complex decision boundaries appear.
6. [multiclass-classifier-mnist.ipynb](notebooks/6.multiclass-classifier-mnist.ipynb)&nbsp;&nbsp;&nbsp;[Video](https://youtu.be/iJt6ZMqbaOo)<br>To demonstrate k class classification instead of binary classification, we use the traditional MNIST digital image recognition problem. We'll again use a random forest model as a baseline classifier. Instead of a sigmoid on a single output neuron, k class classifiers use k neurons in the final layer and then a softmax computation instead of a simple sigmoid. We see fairly decent recognition results with just 50 neurons.  By using 500 neurons, we get slightly better results.
7. [gpu-mnist.ipynb](notebooks/7.gpu-mnist.ipynb)&nbsp;&nbsp;&nbsp;[Video](https://youtu.be/fdBtp7U14zU)<br>This notebook redoes the examples from the previous MNIST notebook but using the GPU to perform matrix algebra in parallel. We use `.to(device)` on tensors and models to shift them to the memory on the GPU. The model trains much faster using the huge number of processors on the GPU. You will need to <a href="https://colab.research.google.com/github/parrt/fundamentals-of-deep-learning/blob/main/notebooks/7.gpu-mnist.ipynb">run the notebook at colab</a> or from an AWS machine to get access to a GPU.
8. [SGD-minibatch-mnist.ipynb](notebooks/8.SGD-minibatch-mnist.ipynb)&nbsp;&nbsp;&nbsp;[Video](https://youtu.be/Rmb1OdXR0eY)<br>We have been doing batch gradient descent, meaning that we compute the loss on the complete training set as a means to update the parameters of the model. If we process the training data in chunks rather than a single batch, we call it mini-batch gradient descent, or more commonly stochastic gradient descent (SGD). It is called stochastic because of the imprecision and, hence, randomness introduced by the computation of gradients on a subset of the training data. We tend to get better generalization with SGD; i.e., smaller validation loss.
9. [data-loaders.ipynb](notebooks/9.data-loaders.ipynb) (optional)&nbsp;&nbsp;&nbsp;[Video](https://youtu.be/4tAZ8zNASSY)<br>PyTorch has support to manage data sets and deal with all of the minibatching. Basically, we will pass in `TensorDataset(X_train, y_train)` and `TensorDataset(X_test, y_test)` instead of `X_train, y_train, X_test, y_test` to our training loop. Then, our loop can iterate through batches using
`(batch_X, batch_y) in train_loader`.

## Part Two

The second part of the course is taught by [Yannet Interian](https://www.usfca.edu/faculty/yannet-interian).

### Notebooks