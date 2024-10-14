
import numpy as np
import cv2
import os
from keras.models import load_model
from PIL import Image
import time

def getImagesAndLabels():
    path = 'dataset'
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

_, ids = getImagesAndLabels()
model = load_model('trained_model.keras')
model.summary()

# Load the label mapping
name_map = {}
original_id_map = {}
with open('Dataset/labels.txt', 'r') as file:
    for line in file:
        parts = line.strip().split(',')
        if len(parts) == 3:
            name, original_id, mapped_id = parts
            name_map[int(mapped_id)] = name
            original_id_map[int(mapped_id)] = int(original_id)

print("Name mapping:")
print(name_map)
print("Original ID mapping:")
print(original_id_map)

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
font = cv2.FONT_HERSHEY_SIMPLEX

def start():
    frame_count = 0
    start_time = time.time()
    fps = 0

    cap = cv2.VideoCapture(0)
    print('Capturing video...')
    ret = True

    while ret:
        ret, frame = cap.read()
        nframe = frame
        faces = faceCascade.detectMultiScale(
            frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        frame_count += 1
        if frame_count >= 30:
            end_time = time.time()
            fps = frame_count / (end_time - start_time)
            frame_count = 0
            start_time = time.time()

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if len(faces) == 0:
            continue

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (32, 32))
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            
            gray = gray.reshape(-1, 32, 32, 1).astype('float32') / 255.
            prediction = model.predict(gray)
            
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)
            
            name = name_map.get(predicted_class, "Unknown")
            original_id = original_id_map.get(predicted_class, "Unknown")
            
            cv2.rectangle(nframe, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(nframe, f"{name} (ID: {original_id})", (x+5, y-5), font, 0.8, (255, 255, 255), 2)
            cv2.putText(nframe, f"Conf: {confidence:.2f}", (x+5, y+h-5), font, 0.5, (255, 255, 0), 1)
        
        cv2.imshow('result', nframe)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

start()