from torchvision import models, transforms
import customtkinter
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

customtkinter.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green
device =  "cuda" if torch.cuda.is_available() else "cpu"
labels = './imagenet_classes.txt'

data = []
labelsTraining = []

def trainieren(categories):
    for category_idx, category in enumerate(categories):
        for file in os.listdir(os.path.join('objekte', category)):
            name, ext = os.path.splitext(file)
            ext = ext[1:]
            if ext == 'npy':
                vector_path = os.path.join('objekte',category, file)
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
        self.objekteList = []
        for file in os.listdir(os.path.join('objekte')):
                self.objekteList.append(file)
        self.modelTrain = None # optional kalo ada kategori kosong bisa tak buang
        
        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)
        
        self.window.geometry(f"{1260}x{680}")
        # self.window.grid_columnconfigure(1, weight=1)
        self.window.grid_columnconfigure((0, 1, 2), weight=0)
        self.window.grid_rowconfigure((0, 1, 2), weight=1)

        self.frame1 = customtkinter.CTkFrame(self.window)
        self.frame1.grid(row=0, column=0,padx=(20, 10), pady=20, sticky="nsew")
        self.logo_label = customtkinter.CTkLabel(self.frame1, text="Objektkategorien", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.canvas = customtkinter.CTkCanvas(self.frame1, width = self.vid.width, height = self.vid.height)
        self.canvas.grid(row=1, column=0, padx=20, pady=10)
        self.label_erkannt= customtkinter.CTkLabel(self.frame1,font=customtkinter.CTkFont(size=14), text='Erkannte Objekt :')
        self.label_erkannt.grid(row =2, column=0, padx=20, pady=(10,5))
        self.text_output = customtkinter.CTkTextbox(self.frame1, width=350, height=100,font=customtkinter.CTkFont(size=12))
        self.text_output.grid(row =3, column=0, padx=20, pady=0, sticky="nsew")

        # Mittel Frame
        self.frame2 = customtkinter.CTkFrame(self.window)
        self.frame2.grid(row = 0,column=1, padx=10, pady=20, sticky="nsew")
        self.label_objektTrainieren = customtkinter.CTkLabel(self.frame2, font=customtkinter.CTkFont(size=18), text='Objekte Trainieren')
        self.label_objektTrainieren.grid(row=0, column=0, padx=20, pady=20)
        self.text_input =customtkinter.CTkEntry(self.frame2, width =250, placeholder_text="Objekt eingeben")
        self.text_input.grid(row=1, column=0,  padx=(20, 20), pady=(0,0), sticky="nsew")
        # Button that lets the user take a snapshot
        self.btn_snapshot=customtkinter.CTkButton(self.frame2, text="Aufnehmen", command=self.buttonClicked)
        self.btn_snapshot.grid(row =2,column = 0,padx=(20,20),pady=(10,20))
        self.label_listedObjects = customtkinter.CTkLabel(self.frame2, font=customtkinter.CTkFont(size=14), text='Aufgenommene- / Ausgewählte Objekte')
        self.label_listedObjects.grid(row=4, column=0, padx=20, pady=(50,10))
        self.btn_objektladen=customtkinter.CTkOptionMenu(self.frame2, values=self.objekteList, command=self.selectPhotoToTrain)
        self.btn_objektladen.set('Objekt laden')
        self.btn_objektladen.grid(row =5,column = 0,padx=(20,20),pady=(0,10))
        self.textEingetragenObjekt = customtkinter.CTkTextbox(self.frame2, width=250, height=80,font=customtkinter.CTkFont(size=12),state =DISABLED)
        self.textEingetragenObjekt.grid(row=6, column=0)
        self.btn_trainieren=customtkinter.CTkButton(self.frame2, text="Trainieren", command=self.buttonTrainieren)
        self.btn_trainieren.grid(row=7, column=0,padx=20,pady=(10,20))

        self.frame3 =customtkinter.CTkFrame(self.window)
        self.frame3.grid(row = 0,column=2, padx=(10,20), pady=20, sticky="nsew")
        self.frame3.grid_rowconfigure(2, weight=0)
        self.label_trainierteObjekt = customtkinter.CTkLabel(self.frame3, font=customtkinter.CTkFont(size=18), text='Neu trainiertes Modell')
        self.label_trainierteObjekt.grid(row=0, column=0, padx=20, pady=20)
        self.btn_laden=customtkinter.CTkButton(self.frame3, text="Modell laden", command=self.browseModelFile)
        self.btn_laden.grid(row =1,column = 0,padx=(20,20),pady=(0,20))
        self.text_output2 = customtkinter.CTkTextbox(self.frame3,width=250, height=70,font=customtkinter.CTkFont(size=12))
        self.text_output2.grid(row =2, column=0, padx=20, pady=0, sticky="nsew")
        self.appearance_mode_label = customtkinter.CTkLabel(self.frame3, text="Erscheinungsmodus :", anchor="s")
        self.appearance_mode_label.grid(row =4, padx=(20, 20), pady=(270, 0))
        self.modeOptionMenu = customtkinter.CTkOptionMenu(self.frame3, values=["Hell", "Dunkel", "System"], command=self.change_appearance_mode)
        self.modeOptionMenu.grid(row =5,padx=20, pady=(10,0))
        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 100
        self.update()

        self.window.mainloop()

    def change_appearance_mode(self, new_appearance_mode: str):
        if new_appearance_mode == 'Hell':
            new_appearance_mode='Light'
        elif new_appearance_mode == 'Dunkel':
            new_appearance_mode ='Dark'
        customtkinter.set_appearance_mode(new_appearance_mode)

    def selectPhotoToTrain(self, ausgewählteObjekt: str):
        # fileTypes = [('Bilddatei', '*.jpg *.png' )]
        # filePath = filedialog.askopenfilename(title='Wählen Bilddatei aus', initialdir='./objekte',filetypes=fileTypes)
        # pfadParts = filePath.split("/")  
        # if not filePath:
        # # User closed the file dialog without selecting any file
        #     messagebox.showerror(title='Fehler', message='Keine Datei ausgewählt.')
        #     return
        # elif pfadParts[5] != 'objekte':
        #     messagebox.showerror(title='Fehler', message='Falsche Ordner ausgewählt.')
        #     return                 
        if self.modelTrain != None:
                self.modelTrain = None
                self.categories = []
                self.text_output2.delete(1.0,END)
        else:
            pass
        if ausgewählteObjekt not in self.categories:
            self.textEingetragenObjekt.configure(state=NORMAL)
            self.textEingetragenObjekt.insert(1.0,ausgewählteObjekt +'\n')
            self.textEingetragenObjekt.configure(state=DISABLED)
            self.categories.append(ausgewählteObjekt)   
            messagebox.showinfo(title='Ausgewählte Datei',message='Sie haben Objekt '+ ausgewählteObjekt +' ausgewählt')
        else: 
            messagebox.showinfo(title='Ausgewählte Datei',message='Sie haben das Objekt bereit '+ ausgewählteObjekt +' ausgewählt')
        
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
        messagebox.showinfo(title='Ausgewählte Datei',message='Sie haben '+ ausgewählteDatei +' Model ausgewählt')
        self.modelTrain = pickle.load(open(f'./model/{ausgewählteDatei}','rb'))
        ausgewählteDateiOhneExt = os.path.splitext(ausgewählteDatei)[0]
        parts = ausgewählteDateiOhneExt.split('-')
        for i in parts[1:]:
            self.categories.append(i)
        #     print('nachher' ,self.categories)
        # print('end',self.categories)

    def checkCategory(self):
        if len(self.categories) != 0:
            self.textEingetragenObjekt.configure(state=NORMAL)
            self.textEingetragenObjekt.delete(1.0,END)
            self.textEingetragenObjekt.configure(state=DISABLED)
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
        else:
            self.makeSnapshot = False
            messagebox.showinfo("Information", "Bilder erfolgreich gespeichert!")
            self.text_input.delete(0, END)
            self.btn_snapshot.configure(text="Aufnehmen")
    
    def buttonTrainieren(self):
        if len(self.categories) > 1:
            if self.modelTrain == None:
                trainieren(self.categories)
                self.textEingetragenObjekt.configure(state=NORMAL)
                self.textEingetragenObjekt.delete(1.0,END)
                self.textEingetragenObjekt.configure(state=DISABLED)
                self.modelTrain = pickle.load(open(f'./model/{fileName}' + '.p','rb'))
            else:
                messagebox.showinfo('Information', 'Ein Model ist bereit trainiert! Um neue Modelle zu trainieren, bitte zuerst neue Objekte erfassen / auswählen')
        elif len(self.categories) == 1:
            messagebox.showerror('Error', 'Die Anzahl der Klassen muss größer als eins sein!\n Bitte noch weitere Objekte aufnehmen / auswählen')
        else:
            messagebox.showerror('Error', 'Kein neue Objekte gegeben \n Bitte Objekte trainieren')
        self.objekteList = []
        for file in os.listdir(os.path.join('objekte')):
                self.objekteList.append(file)
        self.btn_objektladen.configure(values=self.objekteList)
        self.btn_objektladen.set('Objekt laden')

    def snapshot(self):
        if self.makeSnapshot == True:
            # Check if Entry is empty
            entry_value = self.text_input.get()
            if len(entry_value)==0:
                messagebox.showerror('Error', 'Text Eingabe ist leer!')
                self.makeSnapshot = False
                return
            elif entry_value.isspace:
                messagebox.showerror('Error', 'Es ist nur Leerzeichen! Bitte Buchstabe eingeben')
                self.makeSnapshot = False
                return
            else: 
                pass
            self.btn_snapshot.configure(text="Stop")
            # Get a frame from the video source
            ret, frame,vector = self.vid.get_beide()
            self.folderName = self.text_input.get()
            if not os.path.isdir(f'objekte/{self.folderName}'):
                os.mkdir(f'objekte/{self.folderName}')
            if ret:
                cv2.imwrite(f'objekte/{self.folderName}/frame' + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))#
                vector = vector.detach().numpy()
                numpy.save(f'objekte/{self.folderName}/array' + time.strftime("%d-%m-%Y-%H-%M-%S"), vector)
            if self.folderName not in self.categories:
                self.textEingetragenObjekt.configure(state=NORMAL)
                self.textEingetragenObjekt.insert(1.0,self.folderName+'\n')
                self.textEingetragenObjekt.configure(state=DISABLED)
                self.categories.append(self.folderName)
            self.window.after(1000, self.snapshot)
        

    def update(self):
    # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        if ret:
            # For frame
            self.photo = ImageTk.PhotoImage(image = Image.fromarray(frame))
            self.canvas.create_image(0,0,image = self.photo, anchor ='nw')
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
            self.window.after(self.delay, self.update)

class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        self.vid.set(3, 848)
        self.vid.set(4, 540)
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
App(customtkinter.CTk(), "Objektkategorien")