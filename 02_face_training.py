
import cv2
import numpy as np
from PIL import Image
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Activation, Flatten
from keras.utils import to_categorical
from keras import backend as K 
from sklearn.model_selection import train_test_split
from keras import callbacks
from Model import model

# Path for face image database
path = 'dataset'

def downsample_image(img):
    img = Image.fromarray(img.astype('uint8'), 'L')
    img = img.resize((32,32), Image.LANCZOS)
    return np.array(img)

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        try:
            PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
            img_numpy = np.array(PIL_img,'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faceSamples.append(img_numpy)
            ids.append(id)
        except:
            continue    

    return faceSamples, ids

print("\n [INFO] Training faces now.")
faces, ids = getImagesAndLabels(path)

K.clear_session()

# Get the actual number of unique faces
unique_ids = np.unique(ids)
n_faces = len(unique_ids)

# Create a mapping of original IDs to consecutive integers starting from 0
id_map = {id: index for index, id in enumerate(sorted(unique_ids))}

# Map the original IDs to new consecutive IDs
mapped_ids = np.array([id_map[id] for id in ids])

model = model((32,32,1), n_faces)
faces = np.array([downsample_image(face) for face in faces])
faces = faces[:,:,:,np.newaxis]
print("Shape of Data: " + str(faces.shape))
print("Number of unique faces: " + str(n_faces))

# Convert mapped_ids to categorical
ids_categorical = to_categorical(mapped_ids, num_classes=n_faces)

faces = faces.astype('float32')
faces /= 255.

x_train, x_test, y_train, y_test = train_test_split(faces, ids_categorical, test_size=0.2, random_state=0)

checkpoint = callbacks.ModelCheckpoint('trained_model.keras',
                                       save_best_only=True, verbose=1)

model.fit(x_train, y_train,
          batch_size=32,
          epochs=10,
          validation_data=(x_test, y_test),
          shuffle=True, callbacks=[checkpoint])

# Save the final model
model.save('trained_model.keras')

print("\n [INFO] " + str(n_faces) + " faces trained. Exiting Program")

# Read existing labels
existing_labels = []
if os.path.exists('Dataset/labels.txt'):
    with open('Dataset/labels.txt', 'r') as f:
        existing_labels = f.read().splitlines()

# Update or append to the labels
with open('Dataset/labels.txt', 'w') as f:
    for original_id, mapped_id in id_map.items():
        if mapped_id < len(existing_labels):
            name = existing_labels[mapped_id].split(',')[0] if ',' in existing_labels[mapped_id] else existing_labels[mapped_id]
            f.write(f"{name},{original_id},{mapped_id}\n")
        else:
            f.write(f"Person {original_id},{original_id},{mapped_id}\n")

print("\n [INFO] Updated label mapping saved to Dataset/labels.txt")