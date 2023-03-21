# Objektkategorien Application

An application that utilizes webcam input to categorize objects in real-time.


## About

This is a desktop application programmed in Python that uses a pre-trained ImageNet model to categorize objects. Additionally, the application allows users to train a new model to categorize new objects.

![image](/uploads/d6a34d8977c6290af9d0fc845959d3e8/image.png)


## What is in this repository?
In the `/objects/` directory, you can find all the numpy arrays and images for each new object that has already been captured <br />
In the `/model/` directory, you can see all trained models. The names of the models indicate which objects they represent <br />
In the `/label/` directory, you can find the ImageNet classes and labels


## Below are the tech stacks implemented for the development of the system:

- [Tkinter](https://docs.python.org/3/library/tkinter.html) and [customTkinter](https://github.com/TomSchimansky/CustomTkinter) are used for build the Frontend 
- [PyTorch](https://pytorch.org/get-started/locally/) is also used for access and load the pretrained model
- [scikit-learn](https://scikit-learn.org/stable/install.html) is used for training the new model
- [OpenCV](https://opencv.org/) is also used in Backend to get the connection to the webcam 

## Installation

To use and install, download and unzip the file, then install all the necessary libraries (with pip or conda). After that, you can run the app from `app.js`. For first time please use the new aufnehmen function because the objects and the model are empty.

## Authors and acknowledgment
This is a bachelor's thesis project by Kenneth Austin, supervised by Prof. Dr. Erik Rodner
