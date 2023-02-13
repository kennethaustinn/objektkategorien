import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# prepare data
input_dir = './neueModelle'
categories = ['headset', 'ba']

data = []
labels = []
for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(category)):
        name, ext = os.path.splitext(file)
        ext = ext[1:]
        if ext == 'npy':
            vector_path = os.path.join(input_dir, category, file)
            vector = np.load(vector_path)
            data.append(vector)
            labels.append(category_idx)

x = np.array(data)
y = np.array(labels)

# train / test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=True, stratify=labels)

x_train2D = (x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
x_test2D =(x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
# train classifier
classifier = SVC(probability=True)

parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]

model = GridSearchCV(classifier, parameters)

model.fit(x_train2D, y_train)

y_prediction = model.predict(x_test2D)
# test performance

score = accuracy_score(y_prediction, y_test)

print(f"The model is {score*100}% accurate")

pickle.dump(model, open('./modelNeu.p', 'wb'))