Overview
We implemented a program that uses a webcam to capture your face and determine whether you were smiling or not. We use openCV computer vision library to preprocess the webcam image, and we use logistic regression algorithm to train on a provided dataset and evaluate the new image.


Files:

main.py: the main program. It calls the webcam to take the user's photo, calls the mouth detection function, and creates the logistic regression model

logistic.py: the module containing the implementation of the logistic regression model

mouthdetection.py: given an image, detects the mouth and restricts the image to a rectangle around the mouth

neutral.csv - list of training image filenames with neutral faces in ../data/neutral folder

smiles.csv - list of training image filenames with smiling faces in ../data/smile folder

haarcascade_mouth.xml and haarcascade_frontalface_default.xml: training sets that the openCV library functions need in order to isolate the mouth

the data folder contains our training data. 


Instructions:

The program is written in python. It requires standard packages such as numpy and csv. It also requires the openCV library for python and needs to access the webcam on the computer. The detailed instructions to run the program in VMWare is the following (assuming python is already installed):

You can install the openCV library in Fedora with the command line “$ sudo su -” to go to the root directory, and “$ yum install numpy opencv*”. You will also need to install the Python Imaging Library. To do so, type “sudo pip install Pillow” into the command line. 

The program needs to access the webcam. If the program is run in the appliance and your computer has a webcam, you may need to go to the main menu, under “Virtual Machine/ USB & Bluetooth”, click “connect Webcam”. 

To run the main program, type in the terminal window “$ python main.py”. 

Commands in summary:

$ sudo su -
$ yum install numpy opencv*
$ exit 

$ sudo pip install Pillow

$ python main.py


Algorithm Description
a) OpenCV
The program starts by allowing the user to take a picture of themselves with either a smiling or neutral facial expression using a webcam. Then our program uses an algorithm adopted from the OpenCV library to localize the mouth area.

b) Vectorization
We resize the image such that the output is a 28 pixels by 10 pixels image only containing the person’s mouth and surrounding areas. The images are converted to grayscale and then flattened into a vector of length 280, with each entry representing the grayscale of a pixel.

c) Logistic Regression
We built a logistic regression program that will take the user-­provided image vector and determine whether that person was smiling or not. First, the logistic regression is built to take an input of dimension 280. It applies a set of weights to that input and then yields a single scalar. Whether the activation is closer to 0 or 1 determines whether the model will say that the original person was smiling or not.
Before the logistic regression is able to classify the user­-provided image, we trained the model using gradient descent. We used 64 neutral images and 43 smiling images from online datasets in order to train the model and fine­tune the weights. With the suitable weights and biases, we can input the user’s processed mouth image into the model and the network can predict whether that person was smiling or not. The program prints either “You are smiling!” or “You are not smiling!”


NOTE: For the program to work, the user who is taking the image must be in a well­lit area, must be front and center, and must smile fairly widely (i.e. showing teeth).

Datasets:
http://mplab.ucsd.edu, The MPLab GENKI Database. http://pics.psych.stir.ac.uk/2D_face_sets.htm. Utrecht EVP

