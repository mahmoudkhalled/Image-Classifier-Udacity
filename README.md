Image Classifier (Summer 2018)

Summary

With the help of Udacity’s AI Programming with Python Nanodegree program, I built and trained a deep neural network to develop an image classifier for different kinds of flowers in the 2 “image_classifier” files (both are the same code, just different file formats).The second portion of the project includes the two files “predict.py” and “train.py” which allow the user to enter hyperparameters and pre-trained networks types into the command line to train and predict a dataset of images. Both parts print training loss, validation loss, and testing accuracy.

About Project Part I - Deep Learning Model

In a Jupyter Notebook, a flower dataset of images is downloaded as a training, testing, and validation set. Then the datasets are transformed in order to increase accuracy as well as fit the input format for pre-trained networks. Resizing, cropping, random flipping are a few transformations.
Next, vgg16 is chosen to use as the pre-trained network, and a NeuralNetwork class with a feedforward method is defined. Both ReLU activation and dropout are used in the the classifier. After defining hyperparameters, such as number of epochs, the learning rate, etc. the model is trained on the training set. Training loss, validation loss, and accuracy are printed. After running the test set through the model, an accuracy of about 79% is achieved.
A checkpoint saves the model, classifier, and its hyperparameters. A predict and check function are defined which output the top 5 possible flower species for a given image, along with their probabilities in a bar chart.

About Project Part II - Command Line Application

The second portion of the project includes the 2 python files, “train.py” and “predict.py.” This application allows people to train a model on a dataset of images and then predict their classes from the command line. The train file uses the same NeuralNetwork class from Part I, but now the user can choose either vgg16, alexnet, or resnet18 as the pre-trained network. Other parameters, such as number of epochs, number of hidden layers, etc. can be changed from the user. This file should output training loss, validation loss, and accuracy; as well as save a checkpoint. In the predict file, the checkpoint from the train file is loaded and then the top ‘k’ classes and their probabilities are printed.

Getting Started - Running the Files

All files were written using Python. However, part I of the project was written and tested in a Jupyter notebook (where there is also Markdown).

To run the “train.py” file, the only mandatory argument will be the data directory of the images to train. However, one can further control these parameters (which already have defaults): epochs, learning rate, to use GPU or CPU to train, number of hidden layers, and one of three retrained models (vgg16, resnet18, alexnet).
	
	Example: python train.py /path/to/data/directory —arch alexnet —epochs 3
	
“predict.py” follows the same format except there are different parameters in control of the user. Also, “train.py” saves a checkpoint, so the 2 mandatory arguments for “predict.py” are the image to predicted and the checkpoint file containing the trained model.
	
	Example: python predict.py /path/to/image checkpoint.file —category_names path/to/file/containing/category/names

The “image_classifier” files include a dataset of flowers downloaded and transformed to be then trained, predicted, and evaluated. No need to input arguments into this file.

Built With

	•	Python, PyTorch
	•	Pandas, Numpy, Matplotlib
	•	Jupyter Notebook

Contributing

This project was guided by Udacity’s Nanodegree Program: AI Programming with Python.

Authors

Christine Garver

