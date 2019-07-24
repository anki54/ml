#!/usr/bin/env python
# coding: utf-8

# In[3]:


import keras
from matplotlib import pyplot as plt
import numpy as np
import gzip
get_ipython().run_line_magic('matplotlib', 'inline')
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau,CSVLogger
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
from keras.utils import to_categorical
from numpy import array
import pickle
from sklearn.model_selection import train_test_split
import pandas as pd     


# In[2]:


K.tensorflow_backend._get_available_gpus()


# In[4]:


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


# 


# In[5]:


# number of samples to take in each class
N = 1000

# number of epochs/iterations
N_EPOCHS = 5

N_classes = len(LABELS)
classes = {}
files=[]
for i in range(0,N_classes):
  classes.update({i:LABELS[i]})
  files.append(LABELS[i]+".npy")


# In[7]:


def normalize_image(data):
    return np.interp(data, [0, 255], [0, 1])

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

def make_labels(N1, N2):
    labels = []
    for i in range(N1):
        labels += [i] * N2
    return labels

#Takes in a list of filenames and returns a list of numpy arrays
def load_google_data(dir, reshaped, files,N):

    m=0
    data = []
    for file in files:
        f = np.load(dir + file)
        if reshaped:
            new_f = []
            for i in range(N):
                x = np.reshape(f[i], (28, 28))
                x = np.expand_dims(x, axis=0)
                x = np.reshape(f[i], (28, 28, 1))
                new_f.append(x)
            f = new_f
        data.append(f)
        m+=1
        print("loaded file" , m)
    return data


# #data already loaded by CNN script

# In[9]:


with open('finaldatalist', 'rb') as fp:
    data = pickle.load(fp)


# In[10]:


len(data)


# In[11]:


train_data=array(data)


# In[ ]:





# In[12]:


train_data=train_data.reshape(N*N_classes,28,28,1)
print(train_data.shape)


# In[13]:


y_trn=[int(i/N) for i in range(N*N_classes) ]
print(len(y_trn))


# In[14]:


train_data = train_data / np.max(train_data)


# In[15]:


train_X,valid_X,train_ground,valid_ground = train_test_split(train_data,
                                                             train_data,
                                                             test_size=0.2,
                                                             random_state=13)


# In[16]:


batch_size = 64
epochs =N_EPOCHS
inChannel = 1
x, y = 28, 28
input_img = Input(shape = (x, y, inChannel))
num_classes = 345


# In[17]:


## this is the AutoEncode Decoder which will learn weights to be used later in final CNN. 

def encoder(input_img):
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 256 (small and thick)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    return conv4

def decoder(conv4):    
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4) #7 x 7 x 128
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5) #7 x 7 x 64
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    up1 = UpSampling2D((2,2))(conv6) #14 x 14 x 64
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 32
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    up2 = UpSampling2D((2,2))(conv7) # 28 x 28 x 32
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
    return decoded


# In[18]:


autoencoder = Model(input_img, decoder(encoder(input_img)))
autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())
autoencoder.summary()


# In[19]:


callback = [EarlyStopping(patience=1),
                ReduceLROnPlateau(patience=5, verbose=1),
                CSVLogger('autoencoder1log.csv'),
                ModelCheckpoint('autoencoder1.check',
                                save_best_only=True,
                                save_weights_only=True)]


# In[21]:


autoencoder_train = autoencoder.fit(train_X, train_ground, 
                                    batch_size=batch_size,
                                    epochs=2,
                                    verbose=1,validation_data=(valid_X, valid_ground),
                                   callbacks=callback)


# In[27]:


autoencoder.save_weights('autoencoder.h5')


# In[23]:


autoencoder.load_weights('autoencoder.h5')


# In[24]:


# using one-hot encoding
train_labels=[int(i/N) for i in range(0,N*N_classes) ]
print(len(train_labels))
train_Y_one_hot = to_categorical(train_labels)


# In[25]:


train_X,valid_X,train_label,valid_label = train_test_split(train_data,train_Y_one_hot,test_size=0.2,random_state=13)


# In[26]:


train_X.shape,valid_X.shape,train_label.shape,valid_label.shape


# In[27]:


def encoder(input_img):
   #encoder
   #input = 28 x 28 x 1 (wide and thin)
   conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
   conv1 = BatchNormalization()(conv1)
   conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
   conv1 = BatchNormalization()(conv1)
   pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
   conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
   conv2 = BatchNormalization()(conv2)
   conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
   conv2 = BatchNormalization()(conv2)
   pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
   conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)
   conv3 = BatchNormalization()(conv3)
   conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
   conv3 = BatchNormalization()(conv3)
   conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 256 (small and thick)
   conv4 = BatchNormalization()(conv4)
   conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
   conv4 = BatchNormalization()(conv4)
   return conv4


# In[28]:


def fc(enco):
    flat = Flatten()(enco)
    den = Dense(128, activation='relu')(flat)
    out = Dense(num_classes, activation='softmax')(den)
    return out


# In[29]:


## CNN model which will use the weights from AutoEncoder Decoder
encode = encoder(input_img)
final_model = Model(input_img,fc(encode))


# In[30]:


autoencoder.load_weights('autoencoder.h5')


# In[31]:


for l1,l2 in zip(final_model.layers[:19],autoencoder.layers[0:19]):
    l1.set_weights(l2.get_weights())


# Training the encoder only by freezing all the layers except for the softmax. (Equivalent to forward prop)

# In[32]:


for layer in final_model.layers[0:19]:
    layer.trainable = False


# In[33]:


final_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])


# In[34]:


final_model.summary()


# In[35]:


callback = [EarlyStopping(patience=1),
                ReduceLROnPlateau(patience=5, verbose=1),
                CSVLogger('autoencoder2log.csv'),
                ModelCheckpoint('autoencoder2.check',
                                save_best_only=True,
                                save_weights_only=True)]


# In[ ]:


final_model_history = final_model.fit(train_X, train_label, batch_size=64,epochs=10,verbose=1,validation_data=(valid_X, valid_label),
                               callbacks=callback)


# In[ ]:


final_model.save_weights('autoencoder_classification_10.h5')


# Now training the encoder by unfreezing all the layers

# In[37]:


for layer in final_model.layers[0:19]:
    layer.trainable = True


# In[38]:


final_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])


# In[39]:


callback = [EarlyStopping(patience=5),
                ReduceLROnPlateau(patience=5, verbose=1),
                CSVLogger('autoencoder3log.csv'),
                ModelCheckpoint('autoencoder3.check',
                                save_best_only=True,
                                save_weights_only=True)]


# In[45]:


final_model_his_2 = final_model.fit(train_X, train_label, batch_size=64,epochs=5,verbose=1,validation_data=(valid_X, valid_label),
                                callbacks=callback)


# In[49]:


final_model.save_weights('classification_complete.h5')


# In[52]:


from keras.models import load_model
with open('classification_complete.h5', 'rb') as fp:
    final_model=load_model(fp)


# In[53]:


final_model.load_weights("classification_complete.h5")


# Testing on random 10 images of each classes

# In[41]:


with open('test_final', 'rb') as ft:
    testlist = pickle.load(ft)


# In[45]:


testlist = list(map(normalize_image, testlist))


# In[46]:


test_labels = make_labels(345, 10)
test_Y = keras.utils.to_categorical(test_labels, 345)


# In[47]:


testlist = set_limit(testlist, N)


# In[56]:


len(testlist)


# In[54]:


predsTest = final_model.predict(np.array(testlist))
# print(predsTest)
score = 0
for i in range(len(predsTest)):
    #print(np.argmax(predsTest[i]))
    if np.argmax(predsTest[i]) == test_labels[i]:
        score += 1

print("Accuracy: ", ((score + 0.0) / len(predsTest)) * 100,"%")


# In[ ]:




