#!/usr/bin/env python
# coding: utf-8

# # Doodle Classifier Using CNN

# In[1]:


from sklearn.model_selection import train_test_split as tts
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.utils import np_utils
from random import randint
import numpy as np
import os
from PIL import Image
import pickle
from matplotlib import pyplot as plt
from keras import backend as K
K.tensorflow_backend._get_available_gpus()


# available labels- 345 

# In[2]:


LABELS = np.array(['aircraft carrier','airplane','alarm clock','ambulance','angel',
                   'animal migration','ant','anvil','apple','arm','asparagus','axe',
                   'backpack','banana','bandage','barn','baseball','baseball bat',
                   'basket','basketball','bat','bathtub','beach','bear','beard','bed',
                   'bee','belt','bench','bicycle','binoculars','bird','birthday cake',
                   'blackberry','blueberry','book','boomerang','bottlecap','bowtie','bracelet',
                   'brain','bread','bridge','broccoli','broom','bucket','bulldozer','bus','bush',
                   'butterfly','cactus','cake','calculator','calendar','camel','camera','camouflage','campfire','candle','cannon','canoe','car','carrot','castle','cat','ceiling fan','cello','cell phone','chair',
                   'chandelier','church','circle','clarinet','clock','cloud','coffee cup','compass','computer','cookie','cooler','couch','cow','crab','crayon','crocodile','crown','cruise ship','cup','diamond',
                   'dishwasher','diving board','dog','dolphin','donut','door','dragon','dresser','drill','drums','duck','dumbbell','ear','elbow','elephant','envelope','eraser','eye','eyeglasses','face','fan',
                   'feather','fence','finger','fire hydrant','fireplace','firetruck','fish','flamingo','flashlight','flip flops','floor lamp','flower','flying saucer','foot','fork','frog','frying pan','garden',
                   'garden hose','giraffe','goatee','golf club','grapes','grass','guitar','hamburger','hammer','hand','harp','hat','headphones','hedgehog','helicopter','helmet','hexagon','hockey puck','hockey stick',
                   'horse','hospital','hot air balloon','hot dog','hot tub','hourglass','house','house plant','hurricane','ice cream','jacket','jail','kangaroo','key','keyboard','knee','knife','ladder','lantern','laptop',
                   'leaf','leg','light bulb','lighter','lighthouse','lightning','line','lion','lipstick','lobster','lollipop','mailbox','map','marker','matches','megaphone','mermaid','microphone','microwave','monkey',
                   'moon','mosquito','motorbike','mountain','mouse','moustache','mouth','mug','mushroom','nail','necklace','nose','ocean','octagon','octopus','onion','oven','owl','paintbrush','paint can','palm tree',
                   'panda','pants','paper clip','parachute','parrot','passport','peanut','pear','peas','pencil','penguin','piano','pickup truck','picture frame','pig','pillow','pineapple','pizza','pliers','police car',
                   'pond','pool','popsicle','postcard','potato','power outlet','purse','rabbit','raccoon','radio','rain','rainbow','rake','remote control','rhinoceros','rifle','river','roller coaster','rollerskates',
                   'sailboat','sandwich','saw','saxophone','school bus','scissors','scorpion','screwdriver','sea turtle','see saw','shark','sheep','shoe','shorts','shovel','sink','skateboard','skull','skyscraper',
                   'sleeping bag','smiley face','snail','snake','snorkel','snowflake','snowman','soccer ball','sock','speedboat','spider','spoon','spreadsheet','square','squiggle','squirrel','stairs','star','steak',
                   'stereo','stethoscope','stitches','stop sign','stove','strawberry','streetlight','string bean','submarine','suitcase','sun','swan','sweater','swing set','sword','syringe','table','teapot','teddy-bear',
                   'telephone','television','tennis racquet','tent','The Eiffel Tower','The Great Wall of China','The Mona Lisa','tiger','toaster','toe','toilet','tooth','toothbrush','toothpaste','tornado','tractor',
                   'traffic light','train','tree','triangle','trombone','truck','trumpet','t-shirt','umbrella','underwear','van','vase','violin','washing machine','watermelon','waterslide','whale','wheel','windmill',
                   'wine bottle','wine glass','wristwatch','yoga','zebra','zigzag'])


# In[3]:


# number of samples to take in each class
N = 1000

# number of epochs/iterations
N_EPOCHS = 5

N_LABELS = len(LABELS)
DOODLE = {}
files=[]
for i in range(0,N_LABELS):
  DOODLE.update({i:LABELS[i]})
  files.append(LABELS[i]+".npy")


# In[4]:


#Takes in a list of filenames and returns a list of numpy arrays.
def load_google_data(dir, reshaped, files,start,N):
    data = []
    m=0
    for file in files:
        f = np.load(dir + file)
        if reshaped:
            new_f = []
            for i in range(start,start+N):
                x = np.reshape(f[i], (28, 28))
                x = np.expand_dims(x, axis=0)
                x = np.reshape(f[i], (28, 28, 1))
                new_f.append(x)
            f = new_f
        data.append(f)
        m+=1
        print("loaded file" , m)
    return data


def normalize_images(data):
    return np.interp(data, [0, 255], [-1, 1])


def denormalize_images(data):
    return np.interp(data, [-1, 1], [0, 255])


def visualize_image(array):
    array = np.reshape(array, (28,28))
    img = Image.fromarray(array)
    return img

#Limit elements from each array up to n elements and return a single list
def set_limit(arrays, n):
    new = []
    for array in arrays:
        i = 0
        for item in array:
            if i == n:
                break
            new.append(item)
            i += 1
    return new

#make labels from 0 to N1, each repeated N2 times
def set_labels(N1, N2):
    labels = []
    for i in range(N1):
        labels += [i] * N2
    return labels


# Splitting the load in 2 steps as we were getting out of memory error. 

# In[5]:


inputlist1 = load_google_data("google_data/", True, files[:200],0,N)
with open('outfile1', 'ab') as fp:
    pickle.dump(inputlist1, fp)


# In[6]:


inputlist2 = load_google_data("google_data/", True, files[200:],0,N)
with open('outfile2', 'ab') as fp:
    pickle.dump(inputlist2, fp)


# Load the 2 batches without having to download the data.

# In[6]:


with open('outfile1', 'rb') as f:
    inputlist1 = pickle.load(f)


# In[7]:


with open('outfile2', 'rb') as f:
    inputlist2 = pickle.load(f)


# In[8]:


inputlist = inputlist1+ inputlist2 
with open('finaldatalist', 'ab') as fp:
    pickle.dump(inputlist, fp)


# In[9]:


with open('finaldatalist', 'rb') as f:
    finaldatalist = pickle.load(f)


# In[10]:


visualize_image(finaldatalist[1][1])


# In[11]:


# limit no of samples in each class to N
cnndatalist = set_limit(finaldatalist, N)
# normalize the images
mynewlist = list(map(normalize_images, cnndatalist))
# define the labels
labels = set_labels(N_LABELS, N)


# In[12]:


# prepare the data
x_train, x_test, y_train, y_test = tts(cnndatalist, labels, test_size=0.05)


# In[13]:


# one hot encoding
Y_train = np_utils.to_categorical(y_train, N_LABELS)
Y_test = np_utils.to_categorical(y_test, N_LABELS)


# In[14]:


from keras.layers import Activation, Dense
def custom_cnn(size, conv_layers, dense_layers, conv_dropout=0.2,
                      dense_dropout=0.2):
    model = Sequential()
    model.add( Conv2D(conv_layers[0], kernel_size=(3, 3), padding='same', activation='relu', input_shape=(size, size, 1)) )
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    for conv_layer_size in conv_layers[1:]:
        model.add(Conv2D(conv_layer_size, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        if conv_dropout:
            model.add(Dropout(conv_dropout))

    model.add(Flatten())
    if dense_dropout:
        model.add(Dropout(dense_dropout))

    for dense_layer_size in dense_layers:
        model.add(Dense(dense_layer_size, activation='relu'))
        model.add(Activation('relu'))
        if dense_dropout:
            model.add(Dropout(dense_dropout))

    model.add(Dense(N_LABELS, activation='softmax'))
    return model



# In[29]:


size = 28
model1 = custom_cnn(size=size,
                          conv_layers=[32,64],
                          dense_layers=[128],
                          conv_dropout=False,
                          dense_dropout=0.5 )
model1.summary()


# In[ ]:


model1.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# train
model1_history=model1.fit(np.array(x_train), np.array(Y_train), batch_size=32, epochs=N_EPOCHS)

print("Evaluating model")
preds = model1.predict(np.array(x_test))

score = 0
for i in range(len(preds)):
    if np.argmax(preds[i]) == y_test[i]:
        score += 1

print("Accuracy: ", ((score + 0.0) / len(preds)) * 100)


model1.save("model1"+ ".h5")
print(model1_history.history['loss'])
print("Model1 saved")


# In[15]:


size = 28
model2 = custom_cnn(size=size,
                          conv_layers=[128, 128],
                          dense_layers=[1024],
                          conv_dropout=False,
                          dense_dropout=0.10 )
model2.summary()


# In[ ]:


model2.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# train
model2_history=model2.fit(np.array(x_train), np.array(Y_train), batch_size=32, epochs=N_EPOCHS)

print ("Training complete")

print("Evaluating model")
preds = model2.predict(np.array(x_test))

score = 0
for i in range(len(preds)):
    if np.argmax(preds[i]) == y_test[i]:
        score += 1

print("Accuracy: ", ((score + 0.0) / len(preds)) * 100)


model2.save("model2CNN"+ ".h5")
print(model2_history.history['loss'])
print("Model saved")


# In[18]:


from keras.models import load_model
model_load = load_model('model2CNN.h5')


# Create another list of testing samples other than training and validation set.

# In[34]:


x_test1 = load_google_data("data/", True, files[:200],N,10)
with open('test1', 'ab') as fp:
    pickle.dump(x_test1, fp)


# In[9]:


x_test2 = load_google_data("data/", True, files[200:],N,10)
with open('test2', 'ab') as fp:
    pickle.dump(x_test2, fp)


# In[20]:


with open('test1', 'rb') as fp:
    x_test1 = pickle.load(fp)
with open('test2', 'rb') as fp:
    x_test2 = pickle.load(fp)


# In[21]:


x_test=x_test1+x_test2
with open('test_final', 'ab') as fp:
    pickle.dump(x_test, fp)


# In[22]:


x_test = list(map(normalize_images, x_test))
test_labels = set_labels(N_LABELS, 10)


# In[23]:


x_test = set_limit(x_test, N)


# In[24]:


predsTest = model_load.predict(np.array(x_test))
score = 0
for i in range(len(predsTest)):
    if np.argmax(predsTest[i]) == test_labels[i]:
        score += 1

print("Accuracy: ", ((score + 0.0) / len(predsTest)) * 100)


# In[ ]:




