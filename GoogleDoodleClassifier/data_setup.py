#!/usr/bin/env python
# coding: utf-8

# In[ ]:


mkdir google_data
cd google_data
get_ipython().system('pip install keras')
get_ipython().system('pip install tensorflow')

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


# In[ ]:


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


# In[ ]:


for b in LABELS:
    get_ipython().system("wget 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{b}.npy'")


# In[ ]:


cd ..

