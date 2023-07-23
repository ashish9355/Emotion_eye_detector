import tkinter as tk
from tkinter import filedialog
from tkinter import *
from sklearn import metrics
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Function to load the model
def FacialExpressionModel(json_file,weights_file):
    with open(json_file,"r") as file:
        loaded_model_json =file.read()
        model =model_from_json(loaded_model_json)
    model.load_weights(weights_file)
    model.compile(optimizer="adam",loss='categorical_crossentropy',metrics=["accuracy"])   
    return model

# Initialize GUI
top=tk.Tk() 
top.geometry('800x600')
top.title("Emotion and Eye State Detection")
top.configure(background='#CDCDCD')

# Label setups
label1=Label(top,background="#CDCDCD",font=('arial',15,'bold'))
sign_image=Label(top)

# Classifier and model setups
facec =cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eyec = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
emotion_model = FacialExpressionModel("model_akaggle.json","modelkaggle.h5")
eye_model = FacialExpressionModel("modeleye_a.json","eye_model.h5")

EYE_status = ['Close', 'Open']
Emotions_list=["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Function for emotion detection
def DetectEmotion(file_path):
    global Label_packed
    image=cv2.imread(file_path)
    gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces =facec.detectMultiScale(gray_image,1.3,5)

    for (x, y, w, h) in faces:
        fc=gray_image[y:y+h,x:x+w]
        roi=cv2.resize(fc,(48,48))
        pred=Emotions_list[np.argmax(emotion_model.predict(roi[np.newaxis,:,:,np.newaxis]))]
        print("Predicted Emotion is " + pred)
        label1.configure(foreground="#011638",text="Emotion: " + pred)

# Function for eye closed detection
# Function for eye closed detection
# Function for eye closed detection
def DetectEyeState(file_path):
    global Label_packed
    image = cv2.imread(file_path)
    if image is None:
        print('Error loading image')
        return
    eyes = eyec.detectMultiScale(image, scaleFactor=1.2, minNeighbors=3)

    print(f'{len(eyes)} eyes detected')

    for (x, y, w, h) in eyes:
        fc = image[y:y+h, x:x+w]
        roi = cv2.resize(fc, (80, 80))
        roi = roi / 255
        roi = roi.reshape(-1, 80, 80, 3)
        pred = EYE_status[np.argmax(eye_model.predict(roi))]
        print("Predicted Eye State is " + pred)
        label1.configure(foreground="#011638", text="Eye State: " + pred)

# Function to show both detect buttons
def show_detect_buttons(file_path):
    detect_b1= Button(top,text="Detect Emotion",command=lambda:DetectEmotion(file_path),padx=10,pady=5)
    detect_b1.configure(background="#364156",foreground="white",font=("arial",10,'bold'))
    detect_b1.place(relx=0.79 ,rely=0.46)

    detect_b2= Button(top,text="Detect Eye State",command=lambda:DetectEyeState(file_path),padx=10,pady=5)
    detect_b2.configure(background="#364156",foreground="white",font=("arial",10,'bold'))
    detect_b2.place(relx=0.79 ,rely=0.50)

def upload_image():
    try:
        file_path =filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25,(top.winfo_height()/2.25))))
        im=ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image=im
        label1.configure(text='')
        show_detect_buttons(file_path)
    except:
        pass  

# Upload button setup
upload =Button(top,text="Upload Image",command=upload_image,padx=10,pady=5)
upload.configure(background="#364156",foreground="white",font=("arial",20,'bold'))
upload.pack(side='bottom',pady=50)
sign_image.pack(side='bottom',expand='True')
label1.pack(side='bottom',expand='True')

# Heading setup
heading=Label(top,text="Emotion and Eye State Detector",pady=20,font=("arial",25,'bold'))
heading.configure(background="#CDCDCD",foreground="#364156")
heading.pack()
top.mainloop()