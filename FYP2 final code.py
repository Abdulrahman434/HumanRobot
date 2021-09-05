from sys import exit
import speech_recognition as sr
import pyttsx3
import webbrowser
from datetime import date, timedelta, datetime
import serial  # used to communicate with Arduino board
import pyowm  # used to tell the weather
#from Keys import OPENWEATHER # Keys.py is where I store all my API keys SHANE will use
import operator  # used for math operations
import random  # will be used throughout for random response choices
import os  # used to interact with the computer's directory
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pylab import rcParams
from IPython.display import display, Javascript, Image
from base64 import b64decode, b64encode
import cv2
import numpy as np
import PIL
import io
import html
from time import sleep
from threading import Timer
import time
from keras.models import load_model


rcParams['figure.figsize'] = 20, 10

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import keras
from keras.utils import np_utils
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

model = Sequential()

# Speech Recognition Constants
ArduinoUnoSerial = serial.Serial('com8',9600)
recognizer = sr.Recognizer()
microphone = sr.Microphone()

# Python Text-to-Speech (pyttsx3) Constants
engine = pyttsx3.init()
engine.setProperty('volume', 1.0)
initial_test = 0

# Wake word in Listen Function
#WAKE = getLabel(results[0]).lower()

#Define Datapath
# initialize the Haar Cascade face detection model
face_cascade = cv2.CascadeClassifier("C:\TURBOC3\haarcascade_frontalface_default.xml")

# Used to store user commands for analysis
CONVERSATION_LOG = "Conversation Log.txt"

# Initial analysis of words that would typically require a Google search
SEARCH_WORDS = {"who": "who", "what": "what", "when": "when", "where": "where", "why": "why", "how": "how", "search": "search"}

HAPPY_WORDS = {"happy" , "pleased", "excited", "well", "good", "surprised" , "satisfied", "successful", "succeeded", "Easy"}
SAD_WORDS= {"sad" , "angry", "lost" , "loser" , "bad", "stupid" , "worst","worried", "affraid","fear","alone", "not happy"}
NURAL_WORDS = {"ok", "fear", "normal" ,  "nothing" , "neutral",  "okay" ,"alive","bored","pressure", "tired", "tough",}
BYE_WORDS = {"bye", "see", "soon", "close","thank","salam","leaving", "goodbye"}
Take_photo = {"new picture", "new image", "another picture", "another image", "test my emotion", "new photo", "another photo"}
# Establish serial connection for arduino board
# try:
#     ser = serial.Serial('com3', 9600)
#     LED = True
# except serial.SerialException:
#     print("LEDs are not connected. There will be no lighting support.")
#     # If the LEDs aren't connected this will allow the program to skip the LED commands.
#     LED = False
#     pass

class Bibo:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.counter = 1
        
    def randnum(self, fname):
        lines=open(fname).read().splitlines()
        #print(lines)
        return random.choice(lines) 
    
    def getLabel(self,id):
            return ['ANGRY','DISGUST','FEAR','HAPPY','NEUTRAL','SAD','SURPRISE'][id]
    
    
    def first_emotions (self):
        data_path = 'C:/TURBOC3/jaffe'
        data_dir_list = os.listdir(data_path)
        
        img_rows=256
        img_cols=256
        num_channel=1
        
        num_epoch=10
        
        img_data_list=[]
        
        
        for dataset in data_dir_list:
            img_list=os.listdir(data_path+'/'+ dataset)
            print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
            for img in img_list:
                input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
                #input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
                input_img_resize=cv2.resize(input_img,(128,128))
                img_data_list.append(input_img_resize)
                
        self.img_data = np.array(img_data_list)
        self.img_data = self.img_data.astype('float32')
        self.img_data = self.img_data/255
        self.img_data.shape
        
        global initial_test
        
        if initial_test > 0:
            return self.img_data
        
        num_classes = 7
        
        num_of_samples = self.img_data.shape[0]
        labels = np.ones((num_of_samples,),dtype='int64')
        
        labels[0:29]=0 #30
        labels[30:59]=1 #29
        labels[60:92]=2 #32
        labels[93:124]=3 #31
        labels[125:155]=4 #30
        labels[156:187]=5 #31
        labels[188:]=6 #30
        
        names = ['ANGRY','DISGUST','FEAR','HAPPY','NEUTRAL','SAD','SURPRISE']
        
        
        
        
        
        # convert class labels to on-hot encoding# conve 
        Y = np_utils.to_categorical(labels, num_classes)
        
        #Shuffle the dataset
        x,y = shuffle(self.img_data,Y, random_state=2)
        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=2)
        
        
        # Defining the model
        
        input_shape=self.img_data[0].shape
        
        global model
    
        
        # Feature Extraction
        # model.add(Convolution2D(32,3,3, border_mode='same',input_shape=input_shape))
        # model.add(Activation('relu'))
        # model.add(Convolution2D(32, 3, 3))
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.5))
        
        # model.add(Convolution2D(64, 3, 3))
        # model.add(Activation('relu'))
        # # model.add(Convolution2D(64, 3, 3))
        # # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.5))
        
        # model.add(Convolution2D(128, 3, 3))
        # model.add(Activation('relu'))
        # # model.add(Convolution2D(128, 3, 3))
        # # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.5))
        
        
        model.add(Convolution2D(6, (5, 5), input_shape=input_shape, padding='same', activation = 'relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Convolution2D(16, (5, 5), padding='same', activation = 'relu'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Convolution2D(16, (5, 5), padding='same', activation = 'relu'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Convolution2D(64, (3, 3), activation = 'relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Flatten())
        model.add(Dense(128, activation = 'relu'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation = 'softmax'))
        
        
        # Classification
        # model.add(Flatten())
        # model.add(Dense(64))
        # model.add(Activation('relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(num_classes))
        # model.add(Activation('softmax'))
        
        #Compile Model
        model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=["accuracy"])
        
        
        model.summary()
        model.get_config()
        model.layers[0].get_config()
        model.layers[0].input_shape
        model.layers[0].output_shape
        model.layers[0].get_weights()
        np.shape(model.layers[0].get_weights()[0])
        model.layers[0].trainable
        
        
        from keras import callbacks
        filename='model_train_new.csv'
        filepath="Best-weights-my_model-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5"
        
        csv_log=callbacks.CSVLogger(filename, separator=',', append=False)
        checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [csv_log,checkpoint]
        callbacks_list = [csv_log]
        
        #hist = model.fit(X_train, y_train, batch_size=7, epochs=50, verbose=1, validation_data=(X_test, y_test),callbacks=callbacks_list)
          
          #Model Save
        #model.save_weights('model_weights.h5')
        #model.save('model_keras.h5')
        
        
        # visualizing losses and accuracy
        model = load_model('C:/TURBOC3/SHANE_Digital_Assistant_main/‏‏model_keras_final.h5')
        
     
        
        # Evaluating the model
        score = model.evaluate(X_test, y_test, verbose=0)
        print('Test Loss:', score[0])
        print('Test accuracy:', score[1])
        
        test_image = X_test[0:1]
        print (test_image.shape)
        
        print(model.predict(test_image))
        print(model.predict_classes(test_image))
        print(y_test[0:1])
        
        res = model.predict_classes(X_test[:9])
        plt.figure(figsize=(10, 10))
        
        for i in range(0, 9):
            plt.subplot(330 + 1 + i)
            plt.imshow(X_test[i],cmap=plt.get_cmap('gray'))
            plt.gca().get_xaxis().set_ticks([])
            plt.gca().get_yaxis().set_ticks([])
            plt.ylabel('prediction = %s' % self.getLabel(res[i]), fontsize=14)
        # show the plot
        plt.show()
        
        from sklearn.metrics import confusion_matrix
        results = model.predict_classes(X_test)
        cm = confusion_matrix(np.where(y_test == 1)[1], results)
        plt.matshow(cm)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        return self.img_data

    # Used to hear the commands after the wake word has been said
    def hear(self, recognizer, microphone, response):
        try:
            with microphone as source:
                print("Waiting for command.")
                recognizer.adjust_for_ambient_noise(source)
                recognizer.dynamic_energy_threshold = 3000
                # May reduce the time out in the future
                audio = recognizer.listen(source, timeout=10.0)
                command = recognizer.recognize_google(audio)
                self.remember(command)
                return command.lower()
        except sr.WaitTimeoutError:
            pass
        except sr.UnknownValueError:
            pass
        except sr.RequestError:
            print("Network error.")

    # Used to speak to the user
    def speak(self, text):
        global ArduinoUnoSerial
        ArduinoUnoSerial.write(str.encode('1'))
        time.sleep(1)
        newVoiceRate = 120
        engine.setProperty('rate',newVoiceRate)
        engine.say(text)
        engine.runAndWait()
        ArduinoUnoSerial.write(str.encode('2'))
        time.sleep(1)

    # Used to open the browser or specific folders
    def open_things(self, command):
        # Will need to expand on "open" commands
        if command == "open youtube":
            self.speak("Opening YouTube.")
            webbrowser.open("https://www.youtube.com/channel/UCW34Ghe9-_TCA5Vy3-Agfnw")
            pass

        elif command == "open facebook":
            self.speak("Opening Facebook.")
            webbrowser.open("https://www.facebook.com")
            pass

        elif command == "open my documents":
            self.speak("Opening My Documents.")
            os.startfile("C:/Users/Notebook/Documents")
            pass

        elif command == "open my downloads folder":
            self.speak("Opening your downloads folder.")
            os.startfile("C:/Users/Notebook/Downloads")
            pass

        else:
            self.speak("I don't know how to open that yet.")
            pass

    # Used to track the date of the conversation, may need to add the time in the future
    def start_conversation_log(self):
        today = str(date.today())
        today = today
        with open(CONVERSATION_LOG, "a") as f:
            f.write("Conversation started on: " + today + "\n")

    # Writes each command from the user to the conversation log
    def remember(self, command):
        with open(CONVERSATION_LOG, "a") as f:
            f.write("User: " + command + "\n")

    # Used to answer time/date questions
    def understand_time(self, command):
        today = date.today()
        now = datetime.now()
        if "today" in command:
            self.speak("Today is " + today.strftime("%B") + " " + today.strftime("%d") + ", " + today.strftime("%Y"))

        elif command == "what time is it" or command == "what is the time" :
            self.speak("It is " + now.strftime("%I") + now.strftime("%M") + now.strftime("%p") + ".")

        elif "yesterday" in command:
            date_intent = today - timedelta(days=1)
            return date_intent

        elif "this time last year" in command:
            current_year = today.year

            if current_year % 4 == 0:
                days_in_current_year = 366

            else:
                days_in_current_year = 365
            date_intent = today - timedelta(days=days_in_current_year)
            return date_intent

        elif "last week" in command:
            date_intent = today - timedelta(days=7)
            return date_intent
        else:
            pass


    # If we're doing math, this will return the operand to do math with
    def get_operator(self, op):
        return {
            '+': operator.add,
            '-': operator.sub,
            'x': operator.mul,
            'divided': operator.__truediv__,
            'Mod': operator.mod,
            'mod': operator.mod,
            '^': operator.xor,
                }[op]

    # We'll need a list to perform the math
    def do_math(self, li):
        # passes the second item in our list to get the built-in function operand
        op = self.get_operator(li[1])
        # changes the strings in the list to integers
        int1, int2 = int(li[0]), int(li[2])
        # this uses the operand from the get_operator function against the two intengers
        result = op(int1, int2)
        self.speak(str(int1) + " " + li[1] + " " + str(int2) + " equals " + str(result))

    # Checks "what is" to see if we're doing math
    def what_is_checker(self, command):
        number_list = {"1", "2", "3", "4", "5", "6", "7", "8", "9"}
        # First, we'll make a list a out of the string
        li = list(command.split(" "))
        # Then we'll delete the "what" and "is" from the list
        del li[0:2]

        if li[0] in number_list:
            self.do_math(li)

        elif "what is the date today" in command:
            self.understand_time(command)

        else:
            self.use_search_words(command)

    # Checks the first word in the command to determine if it's a search word
    def use_search_words(self, command):
        self.speak("Here is what I found.")
        webbrowser.open("https://www.google.com/search?q={}".format(command))
        
    def happy_emotion(self, command):
        self.speak(self.randnum('C:/TURBOC3/SHANE_Digital_Assistant_main/happy.txt'))
    
    def sad_emotion(self, command):
        self.speak(self.randnum('C:/TURBOC3/SHANE_Digital_Assistant_main/sad.txt'))
    
    def neutral_emotion(self, command):
        self.speak(self.randnum('C:/TURBOC3/SHANE_Digital_Assistant_main/neutral.txt'))
    
    def bye(self, command):
        try:
            global ArduinoUnoSerial
            
            self.speak("okay, Have A nice day my dear and see you soon")
            self.speak("Just before we leave let's take another picture for you.")
            ArduinoUnoSerial.close()
            global img_data
            pic = self.camera(self.img_data)
            
        finally:
            raise SystemExit

    
    
        
        
        

    # Analyzes the command
    def analyze(self, command):
        try:

            if command.startswith('open'):
                self.open_things(command)
            # USED ONLY FOR YOUTUBE PURPOSES
            # if command == "take over the world":
            #     self.speak("Skynet activated.")
            #     listening_byte = "T"  # T matches the Arduino sketch code for the blinking red color
            #     ser.write(listening_byte.encode("ascii"))  # encodes and sends the serial byte

            elif "introduce yourself" in command:
                self.speak("I am Bibo. I'm a digital assistant and physcolgist as well. what about you?")

            elif command == "what time is it":
                self.understand_time(command)
                
            elif command == "hi" or command == "hello" :
                self.speak("Hi. Hello, How are you")
            
            elif "I am fine" in command or "what about you" in command:
                self.speak("great, I am ")
                

            elif "how are you" in command :
                current_feelings = ["I'm okay. What about you?", "I'm doing well. Thank you. What about you?", "I am doing okay. What about you?"]
                # selects a random choice of greetings
                greeting = random.choice(current_feelings)
                self.speak(greeting)

            elif "what" in command:
                self.what_is_checker(command)
            
            elif any(x in command for x in HAPPY_WORDS):
                self.happy_emotion(command)
            
            elif any(x in command for x in SAD_WORDS):
                self.sad_emotion(command)
            
            elif any(x in command for x in NURAL_WORDS):
                self.neutral_emotion(command)
                
            elif any(x in command for x in BYE_WORDS):
                self.bye(command)
             
            elif "need" in command:
                self.needs(command)
           
            elif any(x in command for x in Take_photo):
                global img_data
                emotion = self.camera(self.img_data)
                self.speak("The new image shows that you are "+emotion+", How do you feel now")
                            
                

            # Keep this at the end
            elif SEARCH_WORDS.get(command.split(' ')[0]) == command.split(' ')[0]:
                self.use_search_words(command)
                

            else:
                self.speak("Sorry, you said," + command +", but I don't know how to do that yet. I might misheard you ")


            listening_byte = "H"  # H matches the Arduino sketch code for the green color
            #ser.write(listening_byte.encode("ascii"))  # encodes and sends the serial byte
        except TypeError:
            print("Warning: You're getting a TypeError somewhere.")
            pass
        except AttributeError:
            print("Warning: You're getting an Attribute Error somewhere.")
            pass

    # Used to listen for the wake word
    def listen(self, recognizer, microphone,WAKE):
        while True:
            try:
                with microphone as source:
                    print("Listening.")
                    recognizer.adjust_for_ambient_noise(source)
                    recognizer.dynamic_energy_threshold = 3000
                    #audio = recognizer.listen(source, timeout=5.0)
                    #response = recognizer.recognize_google(audio)

                    if WAKE == 'happy' or WAKE == 'surprise' :
                        self.speak("From Your face, I can see the happiness  and I am happy for that, Tell me about your feeling.")
                        return WAKE.lower()
                    elif WAKE == 'sad' or WAKE == 'angry' or WAKE == 'disgust':
                        self.speak("From Your face, I can see that you are " + WAKE + ", Tell me what you feel.")
                        return WAKE.lower()
                    elif WAKE == 'neutral' :
                        self.speak("You look fresh, I can see that you are ok, but not happy, Tell me what do you feel.")
                        return WAKE.lower()
                    elif WAKE == 'fear' :
                        self.speak("Your face shows that you are worried from something, How I can help you.")
                        return WAKE.lower()

                        
                        #listening_byte = "L"  # L matches the Arduino sketch code for the blue color
                        #ser.write(listening_byte.encode("ascii"))  # encodes and sends the serial byte
                        #self.speak("How can I help you?")
                        

                    else:
                        pass
            except sr.WaitTimeoutError:
                pass
            except sr.UnknownValueError:
                pass
            except sr.RequestError:
                print("Network error.")
                
    def needs (self, command):
        self.speak("Can You Type in the console what do you need. I will Help you in that")
        answer = None
        timeout = 60
        t = Timer(timeout, print, ['Sorry, times up'])
        t.start()
        prompt = "You have %d seconds to type what you need...\n" % timeout
        answer = input(prompt)
        if answer != None :
            self.speak("I noted What you have typed, I will transfer your comment to the authorised person. and we are going to email you soon, Any thing else? ")
            self.remember(answer)
            return
        
        t.cancel()
        self.speak("Sorry, times up for typing, if you still need something just ask again.")
        return
        
        
        
        
        


# import dependencies

    def camera (self, img_data):
        
        global model

        key = cv2. waitKey(1)
        webcam = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        sleep(2)
        t=0
        while True:
        
            
            try:
                check, frame = webcam.read()
                if check == True:
                
                    cv2.imshow("Capturing", frame) 
                    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)   
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    while (t<10000):
                        t+=1
                    
                     
                    
                    for (x,y,w,h) in faces:
                        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
                        cv2.imwrite(filename='saved_img.jpg', img=frame)
                        cv2.waitKey(5)
                        webcam.release()
                        cv2.destroyAllWindows()
                        break
                    break
                else:
                    continue
                
                #print(check) #prints true as long as the webcam is running
                #print(frame) #prints matrix values of each framecd 
                
              
            
            except(KeyboardInterrupt):
                print("Turning off camera.")
                webcam.release()
                print("Camera off.")
                print("Program ended.")
                cv2.destroyAllWindows()
                break
        
        webcam.release()
        cv2.destroyAllWindows()
            
        testimg_data_list=[]
        test_img=cv2.imread('saved_img.jpg', 1)
            
        test_img_resize=cv2.resize(test_img,(128,128))
        testimg_data_list.append(test_img_resize)
        testimg_data = np.array(testimg_data_list)
        testimg_data = testimg_data.astype('float32')
        testimg_data = testimg_data/255
        testimg_data.shape
        
        print("test image original shaape",testimg_data[0].shape)
        print("image original shaape",img_data[0].shape)
        
        results = np.argmax(model.predict(testimg_data), axis=-1)
        plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB),cmap=plt.get_cmap('Set2'))
        plt.gca().get_xaxis().set_ticks([])
        plt.gca().get_yaxis().set_ticks([])
        plt.xlabel('prediction = %s' % self.getLabel(results[0]), fontsize=25)
        print (self.getLabel(results[0]))
        plt.figure()
        return self.getLabel(results[0])
    
    



def main():
    
    global ArduinoUnoSerial
    
    if(ArduinoUnoSerial.isOpen() == False):
        ArduinoUnoSerial.open()
        
           #Create Serial port object called ArduinoUnoSerialData time.sleep(2)                                                             #wait for 2 secounds for the communication to get established
    print (ArduinoUnoSerial.readline())  
    global initial_test
    s = Bibo()
    
    hearing = 0
    
    img_data = s.first_emotions()
    initial_test +=1
        
    s.start_conversation_log()
    WAKE = s.camera(img_data).lower()
    # Used to prevent people from asking the same thing over and over
    previous_response = ""
    
     
    response = s.listen(recognizer, microphone, WAKE)
    
    
    while True:
        
        hearing += 1
        command = s.hear(recognizer, microphone, response)
        if (command == "" ):
            s.speak("Do you have any thing to say? ")
        
        s.analyze(command)
        previous_response = command

