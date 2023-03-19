# The following code used in App.py for training new model

import glob
import pickle
from tkinter import messagebox
import numpy 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

def train(categories):
    labels_training = []
    data = []
    for category_idx, category in enumerate(categories):
        file_paths = glob.glob(f'./objects/{category}/*.npy')
        for vector_path in file_paths:
            vector = numpy.load(vector_path)
            data.append(vector)
            labels_training.append(category_idx)

    x = numpy.array(data)
    y = numpy.array(labels_training)

    # train / test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=True, stratify=labels_training)

    x_train2D = (x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
    x_test2D =(x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
    # train classifier
    classifier = SVC(probability=True)

    parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]

    model_new = GridSearchCV(classifier, parameters)

    model_new.fit(x_train2D, y_train)
    
    y_prediction = model_new.predict(x_test2D)
    score = accuracy_score(y_prediction, y_test)

    global file_name 
    file_name = 'model-' + '-'.join(categories)
    
    pickle.dump(model_new, open(f'./model/{file_name}' + '.p', 'wb'))
    messagebox.showinfo("Information", f'Objekte erfolgreich trainiert mit {str(round(score*100, 2))}% Genauigkeit!\nModel ist jetzt gespeichert')
