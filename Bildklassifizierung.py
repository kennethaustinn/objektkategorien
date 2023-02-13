from torchvision import models, transforms
from tkinter import *
from tkinter import messagebox
import torch
import torch.nn
from PIL import Image, ImageTk
import time
import os
import numpy
import cv2
import pickle

device =  "cuda" if torch.cuda.is_available() else "cpu"
labels = './imagenet_classes.txt'
modelNeu =pickle.load(open('modelNeu.p','rb'))

data_transforms = transforms.Compose(
[
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),                     
    transforms.Normalize(                    
    mean=[0.485, 0.456, 0.406],                
    std=[0.229, 0.224, 0.225]                  
    )])

def preprocess(image):  
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
        self.label_trainieren=Label(self.window,font="Times 14", text='Bild von dem neuen Objekt schon erfolgreich gespeichert? Dann trainieren')
        self.label_trainieren.pack()
        self.btn_trainieren=Button(window, text="Trainieren", width=25, command=self.buttonTrainieren)
        self.btn_trainieren.pack()
        self.text_output2 = Text(self.window, width=50, height=3,font="Times 12")
        self.text_output2.pack()

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 100
        self.update()

        self.window.mainloop()

    def buttonClicked(self):
        if self.makeSnapshot == False:
            self.makeSnapshot = True
            self.snapshot()
            self.btn_snapshot['text'] = "Stop"
        else:
            self.makeSnapshot = False
            messagebox.showinfo("Information", "Bilder erfolgreich gespeichert!")
            self.text_input.delete(0, END)
            self.btn_snapshot['text'] = "Aufnehmen"
    
    def buttonTrainieren(self):
        categories = []
        a = self.text_input.get()
        categories.append(a)
      

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
            folderName = self.text_input.get()
            if not os.path.isdir(folderName):
                os.mkdir(folderName)
            if ret:
                cv2.imwrite(f'neueModelle/{folderName}/frame' + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))#
                vector = vector.detach().numpy()
                numpy.save(f'neueModelle/{folderName}/array' + time.strftime("%d-%m-%Y-%H-%M-%S"), vector)
                # with open('personal.json', 'w') as json_file:
                #     json.dump(vector, json_file)
            
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
            categories = ['headset', 'ba']
            # img=cv2.imread('./Test/ba/frame09-02-2023-10-04-27.jpg',1)
            # image_data = preprocess(img)
            # logits = model(image_data)
            # vektor = torch.nn.Softmax(dim=-1)(logits)
            vektor = probabilities.detach().numpy()
            probability=modelNeu.predict_proba(vektor)
            for ind,val in enumerate(categories):
                output = (f'{ind+1} {val} : {format(probability[0][ind]*100, ".2f")}%\n')
                self.text_output2.insert(END, output)
            self.text_output2.see(END)
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