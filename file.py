import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import pandas as pd
import cv2
import tkinter as tk
from tkinter import *
from PIL import Image,ImageTk
from datetime import datetime
from tkinter import messagebox, filedialog

# Prepare file paths and labels
X = []
y = []
for root, _, filenames in os.walk('one-piece-cards/Cards'):
    for filename in filenames:
        if filename.endswith('.png'):
            file_path = os.path.join(root, filename)
            X.append(file_path)
            label = os.path.splitext(filename)[0]  # Remove the '.png' extension
            y.append(label)
X = pd.DataFrame(data=X, columns=['File Path'])
y = pd.DataFrame(data=y, columns=['Label'])

# Create a mapping from class index to card name
index_to_card_name = y['Label'].tolist()

# Define the model to match the architecture and class count of the saved model
def load_model(num_classes):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# Path to model weights
model_path = "best_model_notran.pth"

# Set num_classes to the original class count used during training
num_classes = 340  # Update this if the original number of classes differs
model = load_model(num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the model state dict
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
model.eval()

# Define transformation
transform = transforms.Compose([
    transforms.Resize((224, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function for predicting the card name based on an image
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    # Map the predicted index to the card name using the index_to_card_name dictionary
    predicted_card_name = index_to_card_name[predicted.item()]
    return predicted_card_name
##############################################################

###############################################################################
# Defining CreateWidgets() function to create necessary tkinter widgets
def createwidgets():
    root.feedlabel = Label(root, bg="MediumPurple1", fg="white", text="Card Feed", font=('Arial',20))
    root.feedlabel.grid(row=1, column=1, padx=10, pady=10, columnspan=2)

    root.cameraLabel = Label(root, bg="MediumPurple1", borderwidth=3, relief="groove")
    root.cameraLabel.grid(row=2, column=1, padx=10, pady=10, columnspan=2)

    root.feedlabel = Label(root, bg="MediumPurple1", fg="white", text="Card Prediction", font=('Arial',20))
    root.feedlabel.grid(row=1, column=13, padx=10, pady=10, columnspan=2)
#ENTRY
    root.saveLocationEntry = Entry(root, width=30, textvariable=destPath)
    root.saveLocationEntry.grid(row=3, column=1, padx=10, pady=10)
#BROWSE
    root.browseButton = Button(root, width=10, text="BROWSE", command=destBrowse)
    root.browseButton.grid(row=3, column=2, padx=10, pady=10)
#BUTTON
    root.captureBTN = Button(root, text="CAPTURE", command=Capture, bg="MediumPurple2", font=('Arial',15), width=20)
    root.captureBTN.grid(row=4, column=1, padx=10, pady=10)

    root.CAMBTN = Button(root, text="STOP CAMERA", command=StopCAM, bg="MediumPurple2", font=('Arial',15), width=13)
    root.CAMBTN.grid(row=4, column=2)

#PREVIEW OR CAMNERA LABEL WHERE IMAGE GOES

    root.previewlabel = Label(root, bg="MediumPurple1", fg="white", text="CARD PREVIEW", font=('Arial',20))
    root.previewlabel.grid(row=1, column=4, padx=10, pady=10, columnspan=2)

#IAMGE LABEL WHERE IMAGE GOES
    root.imageLabel = Label(root, bg="MediumPurple1", borderwidth=3, relief="groove")
    root.imageLabel.grid(row=2, column=4, padx=10, pady=10, columnspan=2)

#PREDICTION LABEL WHERE IMAGE GOES

    root.predictionLabel = Label(root, bg="MediumPurple1", borderwidth=3, relief="groove")
    root.predictionLabel.grid(row=2, column=12, padx=10, pady=10, columnspan=2)   

    root.openImageEntry = Entry(root, width=30, textvariable=imagePath)
    root.openImageEntry.grid(row=3, column=4, padx=10, pady=10)

    root.openImageButton = Button(root, width=10, text="CARD SEARCH", command=imageBrowse)
    root.openImageButton.grid(row=3, column=5, padx=10, pady=10)

    root.captureBTN = Button(root, text="CHECK CARD", command=showPRED, bg="MediumPurple2", font=('Arial',15), width=20)
    root.captureBTN.grid(row=4, column=13, padx=10, pady=10)


    # Calling ShowFeed() function
    ShowFeed()

# Defining ShowFeed() function to display webcam feed in the cameraLabel;
def ShowFeed():
    # Capturing frame by frame
    ret, frame = root.cap.read()

    if ret:
        # Flipping the frame vertically
        frame = cv2.flip(frame, 1)
        width = int(frame.shape[1] * 75 / 100)
        height = int(frame.shape[0] * 75 / 100)
        new_dimensions = (width, height)
        frame = cv2.resize(frame, new_dimensions, interpolation=cv2.INTER_AREA)
        # Displaying date and time on the feed
        cv2.putText(frame, datetime.now().strftime('%d/%m/%Y %H:%M:%S'), (20,30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,255))

        # Changing the frame color from BGR to RGB
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

        # Creating an image memory from the above frame exporting array interface
        videoImg = Image.fromarray(cv2image)

        # Creating object of PhotoImage() class to display the frame
        imgtk = ImageTk.PhotoImage(image = videoImg)

        # Configuring the label to display the frame
        root.cameraLabel.configure(image=imgtk)

        # Keeping a reference
        root.cameraLabel.imgtk = imgtk

        # Calling the function after 10 milliseconds
        root.cameraLabel.after(10, ShowFeed)
    else:
        # Configuring the label to display the frame
        root.cameraLabel.configure(image='')

def destBrowse():
    # saves image
    destDirectory = filedialog.askdirectory(initialdir="OnePieceModel/one-piece-cards")

    # Displays directory
    destPath.set(destDirectory)

def imageBrowse():
    # makes the path
    global imgName
    
    openDirectory = filedialog.askopenfilename(initialdir="OnePieceModel/one-piece-cards")

    # shows path
    imagePath.set(openDirectory)
    
    image_name = os.path.join(openDirectory)

    # opens image
    imageView = Image.open(openDirectory)

    # Resizing
    imageResize = imageView.resize((300, 300), Image.LANCZOS)

    # makes image object
    imageDisplay = ImageTk.PhotoImage(imageResize)

    imgName = openDirectory

    #makes the label
    root.imageLabel.config(image=imageDisplay)

    # ref
    root.imageLabel.photo = imageDisplay



# Defining Capture() to capture and save the image and display the image in the imageLabel
def Capture():
    global imgName
    image_name = datetime.now().strftime('%d-%m-%Y %H-%M-%S') 

    #IMAGE PATH
    if destPath.get() != '':
        image_path = destPath.get()
    else:
        messagebox.showerror("ERROR", "NO DIRECTORY SELECTED!")

    # making the images png
    imgName = image_path + '/' + image_name + ".png"

    # Capturing the IMAGE
    ret, frame = root.cap.read()
    width = int(frame.shape[1] * 50 / 100)
    height = int(frame.shape[0] * 50 / 100)
    new_dimensions = (width, height)
    frame = cv2.resize(frame, new_dimensions, interpolation=cv2.INTER_AREA)
    # Displaying date and time 
    cv2.putText(frame, datetime.now().strftime('%d/%m/%Y %H:%M:%S'), (430,460), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,255))

    success = cv2.imwrite(imgName, frame)

    # Image open
    saved_image = Image.open(imgName)

    # making image an object
    saved_image = ImageTk.PhotoImage(saved_image)
    # label
    root.imageLabel.config(image=saved_image)
    # ref
    root.imageLabel.photo = saved_image

    # Display
    if success :
        messagebox.showinfo("SUCCESS", "IMAGE CAPTURED AND SAVED IN " + imgName)


def StopCAM():
    # Stopping the camera using release() method of cv2.VideoCapture()
    root.cap.release()

    # Configuring the CAMBTN to display accordingly
    root.CAMBTN.config(text="START CAMERA", command=StartCAM)

    # Displaying text message in the camera label
    root.cameraLabel.config(text="OFF CAM", font=('Arial',70))

def StartCAM():
    # live vid
    root.cap = cv2.VideoCapture(0)

    #WIDTH N HEIGHT
    width_1, height_1 = 640, 480
    root.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width_1)
    root.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height_1)

    #CAMERA STOP
    root.CAMBTN.config(text="STOP CAMERA", command=StopCAM)

    #message removal
    root.cameraLabel.config(text="")

    ShowFeed()

def showPRED():
    
    randomimage = predict(imgName)
    
    
    print(imgName)


    base = randomimage.split('-')[0]
    #change to ur path
    a = os.path.join('/Users/enriquealba/Desktop/Camera/OnePieceModel/one-piece-cards/Cards', base, randomimage + '.png') #OP01-001 

    # opens image
    imageView = Image.open(a)

    # Resizing
    imageResize = imageView.resize((224, 320), Image.LANCZOS)

    # makes image object
    imageDisplay = ImageTk.PhotoImage(imageResize)

    #makes the label
    root.predictionLabel.config(image=imageDisplay)
    root.predictionLabel.photo = imageDisplay


#TKCLASS
root = tk.Tk()

#CAM
root.cap = cv2.VideoCapture(0)

#WIDTH HEIGHT
width, height = 640, 400
root.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
root.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

#TITLE
root.title("Card Cloud")
root.geometry("2000x700")
root.resizable(True, True)
root.configure(background = "thistle1")

#VARIABLE
destPath = StringVar()
imagePath = StringVar()

createwidgets()
root.mainloop()

