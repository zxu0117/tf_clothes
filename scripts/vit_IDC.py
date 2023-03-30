#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import glob
import pandas as pd
import numpy as np
import random
import collections
from vit_keras import vit, utils
import tensorflow_addons as tfa
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.metrics

# In[2]:

IMAGE_SIZE = 64

dataset_loc = '/home/ubuntu/Data/MedImage/'
image_folder = dataset_loc + "IDC_regular_ps50_idx5/"

MODEL_PARAMETERS = {"model_name" : "ViT_Train_3_weights",                    
                    "activation" : "relu", # can be relu / sigmoid - **need to google for binary classification,               
                    "loss" : "binary_crossentropy",
                    "optimizer" : "sgd",
                    "class_mode": "binary",
                    "output_layers" : 1,
                    "output_activation" : "sigmoid",
                    "num_epochs" : 50,
                    "batch_size" : 128}

model_output_folder = dataset_loc + MODEL_PARAMETERS["model_name"] + "/"

### Make model output folder if it doesnt exist
if not os.path.isdir(model_output_folder):
    os.mkdir(model_output_folder)

# Write out parameters
with open(model_output_folder+'MODEL_PARAMETERS.txt','w') as data: 
      data.write(str(MODEL_PARAMETERS))
        
print(MODEL_PARAMETERS)

# In[3]:

'''
file_path = "/Users/suzyxu/Documents/ML/vit/Invasive_Ductal_Carcinoma_Dataset/"

full_paths = []

for f in glob.glob(file_path + "/*/*/*.png", recursive=True):
    full_paths.append(f)

patients = [file.split("/")[7] for file in full_paths]
class_membership = [file.split("/")[8] for file in full_paths]

image_info = {"patient_id": patients, "image_path": full_paths, "class_membership": class_membership}
images_df = pd.DataFrame(image_info)
len(full_paths)
'''

# In[4]:

images_list = []
image_pattern = image_folder + "/**/*.png"
#seen_paths = set()
for image_path in glob.glob(image_pattern, recursive=True):
   # if image_path in seen_paths:
        #continue
   # seen_paths.add(image_path)
    # Extract the filename from the image path
    filename = os.path.basename(image_path)
    # Extract the patient ID from the filename
    patient_id = filename.split("_")[0]
    # Extract the class membership from the image path
    class_membership = filename.split("_")[-1].split(".")[0][-1]
    # Add the image information to the list
    image_info = {"patient_id": patient_id, "image_path": image_path, "class_membership": class_membership}
    images_list.append(image_info)

# Create the DataFrame from the list
images_df = pd.DataFrame(images_list)

#images_df['class_membership'] = images_df['class_membership'].astype(int)

images_df

# In[5]:

'''
random.seed(10)

train_pct = 0.8
test_pct = 0.1
val_pct = 0.1
all_patients = np.unique(images_df[["patient_id"]]).tolist()
num_patients = len(all_patients)

num_train = int(num_patients * train_pct)
num_test = int(num_patients * test_pct)
num_val = num_patients - num_train - num_test

train_set = random.sample(all_patients, num_train)

remaining_patients = [x for x in all_patients if x not in train_set]
len(remaining_patients)

test_set = random.sample(remaining_patients, num_test)
val_set = [x for x in remaining_patients if x not in test_set]
print(val_set)
'''

# In[6]:

train_pct = 0.75
test_pct = 0.15
val_pct = 0.1

all_patients = np.unique(images_df[["patient_id"]]).tolist()
num_patients = len(all_patients)

num_train = int(num_patients * train_pct)
num_test = int(num_patients * test_pct)
num_val = num_patients - num_train - num_test

patient_assignment = []
for i, patient_folder in enumerate(all_patients):
    if i < num_train:
        assignment = "train"
    elif i < num_train + num_test:
        assignment = "test"
    else:
        assignment = "validation"
    patient_assignment.append({"patient_id": os.path.basename(os.path.normpath(patient_folder)), "assignment": assignment})

assignment_df = pd.DataFrame(patient_assignment)
merge_df = pd.merge(images_df, assignment_df, on='patient_id')

merge_df = merge_df.groupby('class_membership').apply(lambda x: x.sample(n=75000)).reset_index(drop = True)

print(collections.Counter(merge_df["assignment"].tolist()))

data_counts = (merge_df.groupby(['patient_id', 'class_membership']).size().reset_index())


# In[7]:

val_data = merge_df[merge_df.assignment == 'validation']
test_data = merge_df[merge_df.assignment == 'test']
train_data = merge_df[merge_df.assignment == 'train']

# In[8]:

data_counts.to_csv(model_output_folder + "data_counts.csv")
merge_df.to_csv(model_output_folder + "full_data.csv")
val_data.to_csv(model_output_folder + "val_data.csv")
test_data.to_csv(model_output_folder + "test_data.csv")
train_data.to_csv(model_output_folder + "train_data.csv")

# In[10]:

model = vit.vit_l32(
    image_size=IMAGE_SIZE,
   # patch_size=patch_size,
    activation=MODEL_PARAMETERS["activation"],
    pretrained=True,
    include_top=False,
    pretrained_top=False,
)

train_datagen = ImageDataGenerator(
        #brightness_range=[0.5, 1.5],
        #rotation_range=.1,
        #width_shift_range=0.15,
        #height_shift_range=0.15,
        rescale=1./255,
        #zoom_range=0.05,
        horizontal_flip=True,
        vertical_flip = True,
        fill_mode='constant')

val_datagen = ImageDataGenerator(rescale=1./255)

# In[12]:

# Keras function just to get the formats correct for keras input, google keras flow from dataframe
train_generator=train_datagen.flow_from_dataframe(
dataframe=train_data,
x_col="image_path",
y_col="class_membership",
batch_size=MODEL_PARAMETERS['batch_size'],
#has_ext=True,
shuffle=True,
class_mode=MODEL_PARAMETERS["class_mode"],
target_size=(IMAGE_SIZE,IMAGE_SIZE))

# In[13]:

# Keras function just to get the formats correct for keras input

val_generator=val_datagen.flow_from_dataframe(
dataframe=test_data,
x_col="image_path",
y_col="class_membership",
batch_size=MODEL_PARAMETERS['batch_size'],
#has_ext=True,
shuffle=False,
class_mode=MODEL_PARAMETERS["class_mode"],
target_size=(IMAGE_SIZE,IMAGE_SIZE))

# In[14]:

x = model.output
predictions = layers.Dense(MODEL_PARAMETERS["output_layers"], activation=MODEL_PARAMETERS["output_activation"])(x)
new_model = Model(inputs=model.input, outputs=[predictions])

# In[15]:

# Json describes the structure of the model, nodes / edges, encoding , normalization etc

model_json = new_model.to_json()
with open(model_output_folder + "ViT_IDC.json", "w") as json_file:
    json_file.write(model_json)
    
# In[16]:

#Monitoring status and writing out based on conditions such as best only true

es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=.5*MODEL_PARAMETERS['num_epochs'])
mc = ModelCheckpoint(
    model_output_folder + "ViT_IDC_Weights.{val_loss:.2f}.h5",
    monitor= "val_loss",
    mode='min',
    verbose=1, 
    save_best_only=True,
    save_weights_only=True,
    save_freq="epoch"
)


# In[17]:

# For each epoch, looks at all images
# Batch means when it will update the loss

nbatches_train, mod = divmod(train_data.shape[0], MODEL_PARAMETERS['batch_size'])
STEP_SIZE_VALID=val_generator.n//val_generator.batch_size

# In[18]:

new_model.compile(
    optimizer=MODEL_PARAMETERS["optimizer"],
    loss=MODEL_PARAMETERS["loss"],
    metrics = [tensorflow.keras.metrics.BinaryAccuracy()])

# In[19]:

history = new_model.fit(
    train_generator,
    validation_data=val_generator,
    validation_steps=STEP_SIZE_VALID,
    steps_per_epoch=nbatches_train,
    epochs=MODEL_PARAMETERS['num_epochs'],
    workers=8,
    shuffle=True,
    verbose=1,
    callbacks=[mc, es])

# In[ ]:
               
with open(model_output_folder+'trainHistoryDict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
        
history.history