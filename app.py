# This is the main class of the application UI is implemented here, 
# as well as all functions and methods.
# In this there are 2 classes App is for implementing the user interface 
# and MyVideo Capture is for implementing the webcam.

import time
import os
import customtkinter
from tkinter import *
from tkinter import messagebox, filedialog
import torch
import torch.nn
from PIL import Image, ImageTk
import numpy
import cv2
import pickle
import training
import pretrained

class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.take_shot = False
        self.categories =[]
        self.list_objects = [file for file in os.listdir('objects')]
        self.new_model = None
        
        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)
        customtkinter.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green
        self.window.geometry(f"{1260}x{680}")
        self.window.grid_columnconfigure((0, 1, 2), weight=0)
        self.window.grid_rowconfigure((0, 1, 2), weight=1)

        # Implementation UI Widgets for the left frame
        self.frame_left = customtkinter.CTkFrame(self.window)
        self.frame_left.grid(row=0, column=0,padx=(20, 10), pady=20, sticky="nsew")
        self.logo_label = customtkinter.CTkLabel(self.frame_left, text="Objektkategorien", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.canvas = customtkinter.CTkCanvas(self.frame_left, width = self.vid.width, height = self.vid.height)
        self.canvas.grid(row=1, column=0, padx=20, pady=10)
        self.label_erkannt= customtkinter.CTkLabel(self.frame_left,font=customtkinter.CTkFont(size=14), text='Erkannte Objekt :')
        self.label_erkannt.grid(row =2, column=0, padx=20, pady=(10,5))
        self.text_output_pre = customtkinter.CTkTextbox(self.frame_left, width=350, height=100,font=customtkinter.CTkFont(size=12))
        self.text_output_pre.grid(row =3, column=0, padx=20, pady=0, sticky="nsew")

        # Implementation UI Widgets for the middle frame
        self.frame_middle = customtkinter.CTkFrame(self.window)
        self.frame_middle.grid(row = 0,column=1, padx=10, pady=20, sticky="nsew")
        self.label_objekt_trainieren = customtkinter.CTkLabel(self.frame_middle, font=customtkinter.CTkFont(size=18), text='Objekte Trainieren')
        self.label_objekt_trainieren.grid(row=0, column=0, padx=20, pady=20)
        self.text_input =customtkinter.CTkEntry(self.frame_middle, width =250, placeholder_text="Objekt eingeben")
        self.text_input.grid(row=1, column=0,  padx=(20, 20), pady=(0,0), sticky="nsew")
        self.button_aufnehmen=customtkinter.CTkButton(self.frame_middle, text="Aufnehmen", command=self.button_clicked)
        self.button_aufnehmen.grid(row =2,column = 0,padx=(20,20),pady=(10,20))
        self.label_gelistete_objekte = customtkinter.CTkLabel(self.frame_middle, font=customtkinter.CTkFont(size=14), text='Aufgenommene- / Ausgewählte Objekte')
        self.label_gelistete_objekte.grid(row=4, column=0, padx=20, pady=(50,10))
        self.button_objekt_laden=customtkinter.CTkOptionMenu(self.frame_middle, values=self.list_objects, command=self.select_photo_to_train)
        self.button_objekt_laden.set('Objekt laden')
        self.button_objekt_laden.grid(row =5,column = 0,padx=(20,20),pady=(0,10))
        self.entered_object = customtkinter.CTkTextbox(self.frame_middle, width=250, height=80,font=customtkinter.CTkFont(size=12),state =DISABLED)
        self.entered_object.grid(row=6, column=0)
        self.button_trainieren=customtkinter.CTkButton(self.frame_middle, text="Trainieren", command=self.training)
        self.button_trainieren.grid(row=7, column=0,padx=20,pady=(10,20))
        
        # Implementation UI Widgets for the right frame
        self.frame_right =customtkinter.CTkFrame(self.window)
        self.frame_right.grid(row = 0,column=2, padx=(10,20), pady=20, sticky="nsew")
        self.frame_right.grid_rowconfigure(2, weight=0)
        self.label_trainierte_objekt = customtkinter.CTkLabel(self.frame_right, font=customtkinter.CTkFont(size=18), text='Neu trainiertes Modell')
        self.label_trainierte_objekt.grid(row=0, column=0, padx=20, pady=20)
        self.button_model_laden=customtkinter.CTkButton(self.frame_right, text="Modell laden", command=self.select_model_file)
        self.button_model_laden.grid(row =1,column = 0,padx=(20,20),pady=(0,20))
        self.text_output_new = customtkinter.CTkTextbox(self.frame_right,width=250, height=70,font=customtkinter.CTkFont(size=12))
        self.text_output_new.grid(row =2, column=0, padx=20, pady=0, sticky="nsew")
        self.appearance_mode_label = customtkinter.CTkLabel(self.frame_right, text="Erscheinungsmodus :", anchor="s")
        self.appearance_mode_label.grid(row =4, padx=(20, 20), pady=(270, 0))
        self.option_menu = customtkinter.CTkOptionMenu(self.frame_right, values=["Hell", "Dunkel", "System"], command=self.change_appearance_mode)
        self.option_menu.grid(row =5,padx=20, pady=(10,0))
        # After it is called once, the update method will be automatically called every delay milliseconds
        self.update()

        self.window.mainloop()

    def change_appearance_mode(self, new_appearance_mode: str):
        # Set up so that the name in german is also understood in the english based system
        if new_appearance_mode == 'Hell':
            new_appearance_mode='Light'
        elif new_appearance_mode == 'Dunkel':
            new_appearance_mode ='Dark'
        customtkinter.set_appearance_mode(new_appearance_mode)

    def select_photo_to_train(self, selected_object: str):   
        # Check if there is model that been loaded or not in the new model text box
        # if there is model then will be deleted         
        if self.new_model != None:
                self.new_model = None
                self.categories = []
                self.text_output_new.delete(1.0,END)
        else:
            pass

        if selected_object not in self.categories:
            self.entered_object.configure(state=NORMAL)
            self.entered_object.insert(1.0,selected_object +'\n')
            self.entered_object.configure(state=DISABLED)
            self.categories.append(selected_object)   
            messagebox.showinfo(title='Ausgewählte Datei',message='Sie haben Objekt '+ selected_object +' ausgewählt')
        else: 
            messagebox.showinfo(title='Ausgewählte Datei',message='Objekt '+ selected_object +' ist bereit ausgewählt')
        
    def select_model_file(self):
        file_types = (('P Datei', '*.p'), ('Alle Datei', '*.*')) 
        file_path = filedialog.askopenfilename(title='Wählen trainierte Modelldatei aus', initialdir='./model',filetypes=file_types)
        self.categories = self.check_category()
        # User closed the file dialog without selecting any file, the loaded model is deleted
        if not file_path:
            messagebox.showerror(title='Fehler', message='Keine Datei ausgewählt')
            self.new_model = None
            self.categories = []
            self.text_output_new.delete(1.0,END)
            return
        # User doesn't select the correct data file, the loaded model is deleted
        if not file_path.endswith('.p'):
            messagebox.showerror(title='Fehler',message='Bitte wählen Sie eine *.p-Datei aus')
            self.new_model = None
            self.categories = []
            self.text_output_new.delete(1.0,END)
            return
        # User selects the correct model data, then the new model is loaded
        selected_file = os.path.basename(file_path)
        messagebox.showinfo(title='Ausgewählte Datei',message='Sie haben '+ selected_file +' Model ausgewählt')
        self.new_model = pickle.load(open(f'./model/{selected_file}','rb'))
        selected_file_without_ext = os.path.splitext(selected_file)[0]
        parts = selected_file_without_ext.split('-')
        for i in parts[1:]:
            self.categories.append(i)
    
    def check_category(self):
        if len(self.categories) != 0:
            self.entered_object.configure(state=NORMAL)
            self.entered_object.delete(1.0,END)
            self.entered_object.configure(state=DISABLED)
            return []
        else: 
            return self.categories
        
    def button_clicked(self):
        if self.take_shot == False:
            self.take_shot = True
            if self.new_model != None:
                self.new_model = None
                self.categories = []
                self.text_output_new.delete(1.0,END)
            else:
                pass
            self.take_photo()
        else:
            self.take_shot = False
            messagebox.showinfo("Information", "Bilder erfolgreich gespeichert!")
            self.text_input.delete(0, END)
            self.button_aufnehmen.configure(text="Aufnehmen")
    
    def training(self):
        # Check if the objects are minimum 2 to train a new model
        if len(self.categories) > 1:
            if self.new_model == None:
                training.train(self.categories)
                self.entered_object.configure(state=NORMAL)
                self.entered_object.delete(1.0,END)
                self.entered_object.configure(state=DISABLED)
                self.new_model = pickle.load(open(f'./model/{training.file_name}' + '.p','rb'))
            else:
                messagebox.showinfo('Information', 'Ein Model ist bereit trainiert! Um neue Modelle zu trainieren, bitte zuerst neue Objekte aufnehmen / auswählen')
        elif len(self.categories) == 1:
            messagebox.showerror('Error', 'Die Anzahl der Klassen muss größer als eins sein!\nBitte noch weitere Objekte aufnehmen / auswählen')
        else:
            messagebox.showerror('Error', 'Kein neue Objekte gegeben \nBitte Objekte aufnehmen / auswählen')
        # Update the drop down menu on objects so the actual captured objects can be loaded for the next training
        self.list_objects = []
        for file in os.listdir('objects'):
                self.list_objects.append(file)
        self.button_objekt_laden.configure(values=self.list_objects)
        self.button_objekt_laden.set('Objekt laden')

    def take_photo(self):
        if self.take_shot == True:
            # Check if entry is empty
            entry_value = self.text_input.get()
            if len(entry_value)==0:
                messagebox.showerror('Error', 'Text Eingabe ist leer!')
                self.take_shot = False
                return
            # Check if entry is only blank space
            elif entry_value.isspace():
                messagebox.showerror('Error', 'Es ist nur Leerzeichen! Bitte gültigen Objektnamen eingeben')
                self.take_shot = False
                return
            else: 
                pass
            self.button_aufnehmen.configure(text="Stop")
            # Get a frame from the video source
            ret, frame, vector = self.vid.get_vector_frame()
            self.folder_name = self.text_input.get()
            if not os.path.isdir(f'objects/{self.folder_name}'):
                os.mkdir(f'objects/{self.folder_name}')
            # Whether the frame is true, it will save the frame as picture and the vector as an numpy array
            if ret:
                cv2.imwrite(f'objects/{self.folder_name}/frame' + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                vector = vector.detach().numpy()
                numpy.save(f'objects/{self.folder_name}/array' + time.strftime("%d-%m-%Y-%H-%M-%S"), vector)

            if self.folder_name not in self.categories:
                self.entered_object.configure(state=NORMAL)
                self.entered_object.insert(1.0,self.folder_name+'\n')
                self.entered_object.configure(state=DISABLED)
                self.categories.append(self.folder_name)
            self.window.after(500, self.take_photo)
        

    def update(self):
        ret, frame = self.vid.get_frame()
        if ret:
            # Load the frame in the UI canvas
            self.photo = ImageTk.PhotoImage(image = Image.fromarray(frame))
            self.canvas.create_image(0,0,image = self.photo, anchor ='nw')
            # Update the result of the pretrained model in text_output_pre
            image_data = pretrained.preprocess(frame)
            logits = pretrained.model(image_data)
            probabilities = torch.nn.Softmax(dim=-1)(logits) 
            sorted_prob_pretrained = torch.argsort(probabilities, dim=-1, descending=True)
            # Show the top 5 objects that match the prediction (pretrained model), 
            # ordered in descending order based on their probability percentage
            for (i, idx) in enumerate(sorted_prob_pretrained[0, :5]):
                    ergebnis=("{}. {}: {:.2f}%\n".format
                    (i+1, pretrained.imagenet_labels[idx.item()].strip(),
                    probabilities[0, idx.item()] * 100))
                    self.text_output_pre.insert(END, ergebnis)
            self.text_output_pre.see(END)
            # Update the result of the new trained model in text_output_new 
            if self.new_model != None:
                vector = probabilities.detach().numpy()
                probability= self.new_model.predict_proba(vector)
                probability_1d = probability.flatten().tolist()
                dictionary = dict(zip(self.categories, probability_1d))
                sorted_dict = dict(sorted(dictionary.items(), key=lambda item: -item[1]))
                # Show the top 3 objects that match the prediction (new trained model), 
                # ordered in descending order based on their probability percentage
                for idx, (key,value) in enumerate(list(sorted_dict.items())[:3]):
                     output = (f'{idx+1}. {key}: {format(value*100, ".2f")}%\n')
                     self.text_output_new.insert(END, output)
                self.text_output_new.see(END)
            # Call loop in update function to update the webcam and output text with a delay of 100 ms
            self.window.after(100, self.update)

class MyVideoCapture:
    def __init__(self, video_source=0, width=848, height=480):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.vid.isOpened():
            raise ValueError("Videoquelle kann nicht geöffnet werden", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        # Check if the video or webcam is sucessfuly opened and send back the frame
        # if its true then it will return the  boolean succes flag and the current frame converted to RGB
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    def get_vector_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                image_data = pretrained.preprocess(frame)
                logits = pretrained.model(image_data)
                vector = torch.nn.Softmax(dim=-1)(logits)
            # Return a boolean success flag,the current frame converted to RGB and the vector for training new model
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),vector)
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