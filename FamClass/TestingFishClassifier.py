# TestingFishClassifier.py
# Testing the neural network to do stuff

# source venvfish/bin/activate  # Activate the virtual environment
# cd .. 
# cd FamClass 
# find . -name "*.DS_Store" -type f -delete       # ensures the .DS_Store files generated by mac dont interfere with shit

# python3.9 TestingFishClassifier.py

# Dependancies 
import tensorflow as tf 
import os 
import cv2 
import imghdr 
import numpy as np 
import heapq
from matplotlib import pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy


data_dir = 'FamilyOrdered'

#  Load the model
print('Loading Trained Model')
new_model = load_model(os.path.join('models','famClassModel.h5'))

def FishFamily(imageName): 
    print('Image: ', imageName)
    # Prepping the testing images
    plt.figure()     # new figure
    img = cv2.imread(imageName)
    resize = tf.image.resize(img, (256, 256)) # Transform to put it through the neural network
    plt.imshow(resize.numpy().astype(int))

    # Classifying
    print('Classifying new image')
    yhat_test = new_model.predict(np.expand_dims(resize/255, 0))

    # Get the top 3 predicted Families and print the probability
    yhat_test_edit = yhat_test[0]
    print(yhat_test_edit) # print the probablities
    for index in range(3):
        # print(sorted(os.listdir(data_dir))[np.argmax(yhat_test_edit)])
        print('---- ID Num', np.argmax(yhat_test_edit)+1, ', Family: ', sorted(os.listdir(data_dir))[np.argmax(yhat_test_edit)], ', Confidence: ', yhat_test_edit.max())

        yhat_test_edit[np.argmax(yhat_test_edit)] = 0


   
    


    
if __name__ == "__main__":
    images = ['portjacksonTEST.png','portjacksonTEST2.png', 'blueGroperTEST.png', 'stringRayTEST.png', 'mouriWrassTEST.png', 'TEST4.png']
    imageName = images[5]

    print('Catagories: ', sorted(os.listdir(data_dir)))
    FishFamily(imageName)


# imageName = 'portjacksonTEST.png'


# Halts the program to ensure figures stay viewable with program paused and not ended
plt.show()