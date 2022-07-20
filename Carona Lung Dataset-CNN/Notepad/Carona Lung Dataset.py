#!/usr/bin/env python
# coding: utf-8

# In[1]:


##### Standard Libraries #####
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import sklearn.metrics
import cv2
import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import tensorflow as tf
from glob import glob
from tensorflow import keras
from tensorflow.keras.layers import MaxPooling2D, Conv2D, Flatten, Dense, ZeroPadding2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping


# In[2]:


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images


# In[3]:


import glob
imdir = 'C:/Users/HP-DK0272TX/OneDrive/Desktop/file/Codingrad/Carona Lung set/covid/'
ext = ['jpeg', 'jpg', 'png']    # Add image formats here

files = []
[files.extend(glob.glob(imdir + '*.' + e)) for e in ext]

images = [cv2.imread(file) for file in files]


# In[4]:


len(images)


# In[5]:


norm_images = [cv2.imread(file) for file in glob.glob('C:/Users/HP-DK0272TX/OneDrive/Desktop/file/Codingrad/Carona Lung set/normal/*.jpeg')]


# In[6]:


norm=pd.DataFrame(norm_images,columns=["Images"])


# In[7]:


norm


# In[8]:


cov = pd.DataFrame(images,columns=["Images"])
cov


# # Checking and plot any one image:

# In[9]:


img1=images[0][0][0]
img1


# In[10]:


import matplotlib.pyplot as plt
im = cv2.imread("C:/Users/HP-DK0272TX/OneDrive/Desktop/file/Codingrad/Carona Lung set/covid/01E392EE-69F9-4E33-BFCE-E5C968654078.jpeg", 0) / 255
plt.imshow(im, cmap='gray', vmin=0, vmax=1) 
plt.show()


# # Keras split train test set using Imagedatagenerator:

# In[11]:


from keras.preprocessing import image

train_gen = image.ImageDataGenerator(rescale=1/255,horizontal_flip=True,zoom_range=0.2)

val_gen = image.ImageDataGenerator(rescale = 1/255)


# In[12]:


DIR ='C:/Users/HP-DK0272TX/OneDrive/Desktop/file/Codingrad/Carona Lung set/'
SUBDIR_POS = 'covid/'
SUBDIR_NEG = 'normal/'
print(f'Positive samples: {len(os.listdir(DIR + SUBDIR_POS))}.')
print(f'Negative samples: {len(os.listdir(DIR + SUBDIR_NEG))}.')


# In[13]:


import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

EPOCHS = 40
BATCH_SIZE = 64
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=0.001, decay=0.001 / EPOCHS)
img_height, img_width = 248, 248
es = EarlyStopping(monitor='val_acc', mode='max',
                   verbose=1, 
                   patience=10, restore_best_weights=True)


# In[14]:


#keras-split-train-test-set-when-using-imagedatagenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=5,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
    validation_split=0.2) 

train_generator = train_datagen.flow_from_directory(
    DIR,
    target_size=(img_height, img_width),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    color_mode="grayscale",
    subset='training') 

validation_generator = train_datagen.flow_from_directory(
    DIR, 
    target_size=(img_height, img_width),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    color_mode="grayscale",
    subset='validation')


# In[15]:


train_generator


# In[16]:


validation_generator


# # Function Definition to build CNN Model:

# In[17]:


def create_model():
    model = Sequential([
        Conv2D(16, 1, padding='same', activation='relu', input_shape=(img_height, img_width, 1)),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 5, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 5, padding='same', activation='relu'),
        MaxPooling2D(),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(8, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=OPTIMIZER,
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'Precision', 'Recall'])
    
    return model


# # Model Building

# In[18]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D, Conv2D, Flatten, Dense, ZeroPadding2D


model = create_model()
model.summary()


# In[19]:


hist=model.fit(train_generator, validation_data=validation_generator, epochs=EPOCHS)


# # Plot the loss and accuracy graphs:

# In[20]:


sns.set_style('darkgrid')

plt.title('Accuracy')
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.title('Loss')
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.title('Recall')
plt.plot(hist.history['recall'])
plt.plot(hist.history['val_recall'])
plt.ylabel('recall')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.title('Precision')
plt.plot(hist.history['precision'])
plt.plot(hist.history['val_precision'])
plt.ylabel('recall')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# Testing the model with test data:

# In[21]:


y_pred = (model.predict_generator(validation_generator) > 0.5).astype(int)
y_true = validation_generator.classes

for name, value in zip(model.metrics_names, model.evaluate_generator(validation_generator)):
    print(f'{name}: {value}')
    
print(f'F1 score: {sklearn.metrics.f1_score(y_true, y_pred)}')


# In[22]:


pd.DataFrame(sklearn.metrics.confusion_matrix(y_true, y_pred), 
             columns=['pred no covid', 'pred covid'], 
             index=['true no covid', 'true covid'])


# Save the weight file:

# In[23]:


model_json = model.to_json()
with open("corona_model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")


# In[ ]:




