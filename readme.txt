Files:

main.py: the main program. It calls the webcam to take the user's photo, calls the mouth detection function, and creates the logistic regression model

logistic.py: the module containing the implementation of the logistic regression model

mouthdetection.py: given an image, detects the mouth and restricts the image to a rectangle around the mouth

neutral.csv - list of training image filenames with neutral faces in ../data/neutral folder

smiles.csv - list of training image filenames with smiling faces in ../data/smile folder

haarcascade_mouth.xml and haarcascade_frontalface_default.xml: training sets that the openCV library functions need in order to isolate the mouth

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



