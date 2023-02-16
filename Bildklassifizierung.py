from torchvision import models, transforms
from tkinter import *
from tkinter import messagebox, filedialog
import torch
import torch.nn
from PIL import Image, ImageTk
import time
import os
import numpy
import cv2
import pickle
from trainieren import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

device =  "cuda" if torch.cuda.is_available() else "cpu"
labels = './imagenet_classes.txt'

data = []
labelsTraining = []

def trainieren(categories):
    for category_idx, category in enumerate(categories):
        for file in os.listdir(os.path.join(category)):
            name, ext = os.path.splitext(file)
            ext = ext[1:]
            if ext == 'npy':
                vector_path = os.path.join(category, file)
                vector = numpy.load(vector_path)
                data.append(vector)
                labelsTraining.append(category_idx)

    x = numpy.array(data)
    y = numpy.array(labelsTraining)

    # train / test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=True, stratify=labelsTraining)

    x_train2D = (x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
    x_test2D =(x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
    # train classifier
    classifier = SVC(probability=True)

    parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]

    modelNeu = GridSearchCV(classifier, parameters)

    modelNeu.fit(x_train2D, y_train)

    y_prediction = modelNeu.predict(x_test2D)
    # test performance

    score = accuracy_score(y_prediction, y_test)

    print(f"The model is {score*100}% accurate")

    global fileName 
    fileName = 'modelNeu-' + '-'.join(categories)
    
    pickle.dump(modelNeu, open(f'./model/{fileName}' + '.p', 'wb'))
    
    messagebox.showinfo("Information", f'Objekte erfolgreich trainiert mit {str(score*100)}% Genauigkeit!\n Model ist jetzt gespeichert')

def preprocess(image):  
    data_transforms = transforms.Compose(
    [transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),                     
    transforms.Normalize(                    
    mean=[0.485, 0.456, 0.406],                
    std=[0.229, 0.224, 0.225])])
    image = Image.fromarray(image) #Webcam frames are numpy array format, therefore transform back to PIL image
    image = data_transforms(image)
    image = torch.unsqueeze(image,0)
    return image                            

imagenetLabels = dict(enumerate(open(labels)))
# Load pre trained model
resnet = models.resnet101(pretrained=True)
model = resnet.to(device)  #set where to run the model and matrix calculation
model.eval()

class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.makeSnapshot = False
        self.categories =[]
        self.modelTrain = None # optional kalo ada kategori kosong bisa tak buang
        self.boxOutput2 = False
        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack(side=LEFT)
        self.label_erkannt=Label(self.window,font="Times 14", text='Erkannte Objekt :')
        self.label_erkannt.pack()
        self.text_output = Text(self.window, width=50, height=6,font="Times 12")
        self.text_output.pack()
        self.label_nichterkannt=Label(self.window,font="Times 14", text='Nicht erkannte Objekt? Trainieren :')
        self.label_nichterkannt.pack()
        # Input
        self.text_input =Entry(window, width =50, borderwidth=5)
        self.text_input.pack()
        # Button that lets the user take a snapshot
        self.btn_snapshot=Button(window, text="Aufnehmen", width=25, command=self.buttonClicked)
        self.btn_snapshot.pack()
        self.label_trainieren=Label(self.window,font="Times 14", text='\nBild von dem neuen Objekt schon erfolgreich gespeichert?\nDann trainieren')
        self.label_trainieren.pack()
        self.btn_trainieren=Button(window, text="Trainieren", width=25, command=self.buttonTrainieren)
        self.btn_trainieren.pack()
        self.text_output2 = Text(self.window, width=50, height=4,font="Times 12", borderwidth=5)
        self.text_output2.pack()
        self.btn_laden=Button(window, text="Laden", width=25, command=self.browseModelFile)
        self.btn_laden.pack()
        self.btn_ladenfoto=Button(window, text="Laden Foto", width=25, command=self.selectPhotoToTrain)
        self.btn_ladenfoto.pack()
        self.textEingetragenObjekt = Text(self.window, width=25, height=3,font="Times 12", borderwidth=5)
        self.textEingetragenObjekt.pack()

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 100
        self.update()

        self.window.mainloop()

    def selectPhotoToTrain(self):
        fileTypes = [('Bilddatei', '*.jpg *.png' )]
        filePath = filedialog.askopenfilename(title='Wählen Bilddatei aus', initialdir='./',filetypes=fileTypes)
        pfadParts = filePath.split("/")  
        if not filePath:
        # User closed the file dialog without selecting any file
            messagebox.showerror(title='Fehler', message='Keine Datei ausgewählt.')
            return
        elif pfadParts[4] != 'webcamimagenet':
            messagebox.showerror(title='Fehler', message='Falsche Ordner ausgewählt.')
            return                 
        messagebox.showinfo(title='Ausgewählte Datei',message='Sie haben Objekt '+ pfadParts[5] +' ausgewählt')
        self.categories.append(pfadParts[5])
        self.textEingetragenObjekt.insert(1.0,pfadParts[5] +'\n')
        
    def browseModelFile(self):
        fileTypes = (('P Datei', '*.p'), ('Alle Datei', '*.*')) 
        filePath = filedialog.askopenfilename(title='Wählen trainierte Modelldatei aus', initialdir='./model',filetypes=fileTypes)
        self.categories = self.checkCategory()
        if not filePath:
        # User closed the file dialog without selecting any file
            messagebox.showerror(title='Fehler', message='Keine Datei ausgewählt.')
            self.modelTrain =None
            self.categories = []
            self.text_output2.delete(1.0,END)
            return
        if not filePath.endswith('.p'):
            messagebox.showerror(
                title='Fehler',
                message='Bitte wählen Sie eine *.p-Datei aus.'
            )
            self.modelTrain =None
            self.categories = []
            self.text_output2.delete(1.0,END)
            return
        
        ausgewählteDatei = os.path.basename(filePath)
        messagebox.showinfo(title='Ausgewählte Datei',message='Sie haben '+ ausgewählteDatei +' Modell ausgewählt')
        self.modelTrain = pickle.load(open(f'./model/{ausgewählteDatei}','rb'))
        ausgewählteDateiOhneExt = os.path.splitext(ausgewählteDatei)[0]
        parts = ausgewählteDateiOhneExt.split('-')
        for i in parts[1:]:
            self.categories.append(i)
        #     print('nachher' ,self.categories)
        # print('end',self.categories)

    def checkCategory(self):
        if len(self.categories) != 0:
            return []
        else: 
            return self.categories
        
    def buttonClicked(self):
        if self.makeSnapshot == False:
            self.makeSnapshot = True
            if self.modelTrain != None:
                self.modelTrain = None
                self.categories = []
                self.text_output2.delete(1.0,END)
            else:
                pass
            self.snapshot()
            self.btn_snapshot['text'] = "Stop"
        else:
            self.makeSnapshot = False
            messagebox.showinfo("Information", "Bilder erfolgreich gespeichert!")
            self.text_input.delete(0, END)
            self.btn_snapshot['text'] = "Aufnehmen"
    
    def buttonTrainieren(self):
        if len(self.categories) > 1:
            if self.modelTrain == None:
                trainieren(self.categories)
                self.textEingetragenObjekt.delete(1.0,END)
                self.modelTrain = pickle.load(open(f'./model/{fileName}' + '.p','rb'))
            else:
                messagebox.showinfo('Information', 'Ein Model ist bereit trainiert, um neue Modelle zu trainieren bitte erstmal neue Objekte Aufnehmen')
        elif len(self.categories) == 1:
            messagebox.showerror('Error', 'Die Anzahl der Klassen muss größer als eins sein!\n Bitte noch weitere Objekte trainieren')
        else:
            messagebox.showerror('Error', 'Kein neue Objekte gegeben \n Bitte Objekte trainieren')


    def snapshot(self):
        if self.makeSnapshot == True:
            # Check if Entry is empty
            entry_value = self.text_input.get()
            if len(entry_value)==0:
                messagebox.showerror('Error', 'Text Eingabe ist leer')
                self.makeSnapshot = False
                END
            else:
                pass
            # Get a frame from the video source
            ret, frame,vector = self.vid.get_beide()
            self.folderName = self.text_input.get()
            if not os.path.isdir(self.folderName):
                os.mkdir(self.folderName)
            if ret:
                cv2.imwrite(f'{self.folderName}/frame' + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))#
                vector = vector.detach().numpy()
                numpy.save(f'{self.folderName}/array' + time.strftime("%d-%m-%Y-%H-%M-%S"), vector)
            if self.folderName not in self.categories:
                self.categories.append(self.folderName)
                self.textEingetragenObjekt.insert(1.0,self.folderName+'\n')
            self.window.after(1000, self.snapshot)
        

    def update(self):
    # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        if ret:
            # For frame
            self.photo = ImageTk.PhotoImage(image = Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = NW)
            # for labels 
            image_data = preprocess(frame)
            logits = model(image_data)
            probabilities = torch.nn.Softmax(dim=-1)(logits) 
            # The dim argument is required unless your input tensor is a vector. It specifies the axis along which to apply the softmax activation. 
            # Passing in dim=-1 applies softmax to the last dimension. So, after you do this, the elements of the last dimension will sum to 1.
            sortedProba = torch.argsort(probabilities, dim=-1, descending=True)
            for (i, idx) in enumerate(sortedProba[0, :5]):
                    ergebnis=("{}. {}: {:.2f}%\n".format
                    (i+1, imagenetLabels[idx.item()].strip(),
                    probabilities[0, idx.item()] * 100))
                    self.text_output.insert(END, ergebnis)
            self.text_output.see(END)
            # print('update' + str(self.modelTrain))
            if self.modelTrain != None:
                vektor = probabilities.detach().numpy()
                probability= self.modelTrain.predict_proba(vektor)
                for ind,val in enumerate(self.categories):
                    output = (f'{ind+1}. {val}: {format(probability[0][ind]*100, ".2f")}%\n')
                    self.text_output2.insert(END, output)
                self.text_output2.see(END)
                self.boxOutput2 = True
            self.window.after(self.delay, self.update)

class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
            # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    def get_beide(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                image_data = preprocess(frame)
                logits = model(image_data)
                vektor = torch.nn.Softmax(dim=-1)(logits)
            # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),vektor)
            else:
                return (ret, None, None)
        else:
            return (ret, None, None)
        
        # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
                

# Create a window and pass it to the Application object
App(Tk(), "Objektkategorien")