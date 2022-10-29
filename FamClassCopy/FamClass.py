# ImageClassExample

##################################################################
## 1. Install Dependencies and Setup

# Run in virtual envirnemnt for cv2 to work
# pip install opencv-python matplotlib
# pip list


import tensorflow as tf 
import os 
import cv2 
import imghdr 
import numpy as np 
from matplotlib import pyplot as plt


from tensorflow.keras.applications import VGG16 # pretrained neural network
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential # 2 specific model apis in keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout # droptout not used 
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy



# limits how much of the gpu that can be used by tensorflow. avoids the computer hogging all gpu resources
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

tf.config.list_physical_devices('GPU')


##################################################################
## 2. Remove dodgy images
# import cv2 
# import imghdr 

data_dir = 'FamilyOrdered'

image_ext = ['jpeg','jpg','bmp','png']

for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_ext:
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print('Issue with image {}'.format(image_path))
            # os.remove(image_path)


##################################################################
## 3. Load Data
# Tensorflow has a database API. To access the API directly, eg: tf.data.Dataset??
# Allows you to build a data pipeline instead of directly loading the dataset into memory
# Ensures scalability to larger datasets
# Repeatable set of steps to apply to the data --> cleaner dataset


# import numpy as np 
# from matplotlib import pyplot as plt

# Get the downloaded happy and sad images from the data dir
# keras data pipeline API helper
# automatically resizes the images to 256x256 and catgorises them
data = tf.keras.utils.image_dataset_from_directory(data_dir) # Building the data pipeline
num_direct = len(next(os.walk(data_dir))[1]) # how many catagories are there? 
# print(num_direct)

# return an iterator which converts all elements of the dataset to numpy
# access the generator from our data pipeline --> allow us to loop through data
data_iterator = data.as_numpy_iterator()  

# grabbing one batch 
batch = data_iterator.next()

# Images represented as numpy arrays
# print('****************************************************************************************************')
# print(batch[0])
# print(data.features["labels"].num_classes)
# print('****************************************************************************************************')

# Labels represented as numpy arrays happy [0], sad [1]
# print('****************************************************************************************************')
# print(batch[1])
# print(data.features["labels"].names)
# print('****************************************************************************************************')

# Plot 4 images and title them according to happy [0] or sad [1]
fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])




##################################################################
## 4. Scale Data
# preprocess the image data to between 0 - 1
    # Helps the deep learning model generalise faster and produce better results 

# .map performs the transformation to 0-1 WITHIN the pipeline
# x: images, y: labels
data = data.map(lambda x,y: (x/255, y))
data.as_numpy_iterator().next()

# plot 

##################################################################
## 5. Split the data into its respective training, validation and testing chunks
# Split data into partitions: training, validation, testing to ensure we dont overfit


# sum of train_size, val_size, test_size needs to equal len(data)
train_size = int(len(data)*.7)
val_size = int(len(data)*.2)+1
test_size = int(len(data)*.1)+1

# train_size+val_size+test_size == len(data)

# if not already you need to have shuffled your dataset by now
# .take defines how much data we take in that particular partition
train = data.take(train_size)

# .skip: skip the batches we have already assigned to the training partition
# .take: take 2 batches for the validation partition
val = data.skip(train_size).take(val_size)

# .skip: skip the batches we have already assigned to the training partition
# .take: take remaining 1 batches for the testing partition
test = data.skip(train_size+val_size).take(test_size)


##################################################################
## 6. Build Deep Learning Model
train
# print('****************************************************************************************************')
# print('shape:')
# print(train.shape())
# print('shape 0:')
# print(train.shape[0])
# print('shape 1:')
# print(train.shape[1])
# print('****************************************************************************************************')



## Get useful "functions" out of the packages that we will use for building
# Sequential api is good for models flowing top to bottom --> quick and easy
# functional api multiple inputs multiple outputs 
# from tensorflow.keras.models import Sequential # 2 specific model apis in keras

# Conv2D: spatial convolution over images
# MaxPooling2D: goes over image and condenses them down applies kernal to reduce image size, means that data returned from the convolution is condensed
# Dense: fully connected layer 
# Flatten: allows us to go through a convolutional layer and turns in into a format that Dense can handle
# Dropout: used for regularisation
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout 

# Create a sequential model that will be trained
model = Sequential()
# Can be done like this as well: 
# model = Sequential([Conv2D(), .. ect])

##################################################################

# #  Create the framework

## VGG Download and setup
# vgg = VGG16(include_top=False)
# # Does not used the later layers as we want to pipe into our classifier
# vgg.summary()

# Create a convolution with 16 filters that scans over the image
# Extract the relevant information from that image to make a classification 
# 3pixels x 3pixels
# Have a stride of 1 --> move by 1 pixel each time
# activation: the "neural firing profile" function that remaps the output values to new range
# input_shape: 
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
# runs over the conv2d output an condenses the image data
model.add(MaxPooling2D())

# repeat 1
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
# repeat 2
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

# Flatten down to only 1 dimension 
# max_pooling2d_2 (MaxPooling) dimensions are: (None, 30, 30, 16) 30*30*16=14400 
#  output of flatten is a 1D array of 14400 elements
model.add(Flatten())

# Fully connected layers
model.add(Dense(256, activation='relu'))
# condense to the catagorical outputs (21 classes)
model.add(Dense(num_direct, activation='softmax'))

### Configures the model for training: Compile using the adam optimiser
# tf.losses.BinaryCrossentropy(): for binary classification models 
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.compile(optimizer='adam', loss =tf.keras.losses.SparseCategoricalCrossentropy(), metrics=[tf.keras.metrics.CategoricalAccuracy()])





# , loss=tf.losses.BinaryCrossentropy()

# print out a summary of the model parameters
model.summary()




##################################################################
## 7. Train the model
logdir='logs'
# Log to tensorboard so we can examine the model training later
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])


print('****************************************************************************************************')
print(hist.history.keys())
print('----- loss:', hist.history['loss'])
print('----- val_loss:', hist.history['val_loss'])
print('----- categorical_accuracy:', hist.history['categorical_accuracy'])
print('----- val_categorical_accuracy:', hist.history['val_categorical_accuracy'])
print('****************************************************************************************************')


##################################################################
## 8. Plot Performance
# Plot the loss
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")

# Plot the accuracy
fig2 = plt.figure()     # new figure
plt.plot(hist.history['categorical_accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_categorical_accuracy'], color='orange', label='val_accuracy')
fig2.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
# plt.show()

##################################################################
## 9. Evaluate 
# from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

# Establish instances
pre = Precision()
re = Recall()
acc = BinaryAccuracy()

# Loop through each batch in the testing data
for batch in test.as_numpy_iterator():
    X, y = batch                        # set of images
    # print('----- X:', X) # test images (256,256,3,32)
    # print('----- y:', y) # test labels (32)

    ynew = np.zeros((len(y),num_direct))
    count = 0
    for ylabel in y:
        ynew[count,ylabel] = 1
        count = count + 1

    # Below should output a vector 21 units long
    yhat = model.predict(X)
    # print('----- yhat:', yhat.size)           
    pre.update_state(ynew, yhat)           
    re.update_state(ynew, yhat)            
    acc.update_state(ynew, yhat)  


    
print(f'Precision: {pre.result().numpy()}, Recall: {re.result().numpy()}, Accuracy: {acc.result().numpy()}')

##################################################################
## 10. Test
plt.figure()     # new figure
img = cv2.imread('portjacksonTEST.png')
plt.imshow(img)
# plt.show()

# Transform to put it through the neural network
plt.figure()     # new figure
resize = tf.image.resize(img, (256, 256))
plt.imshow(resize.numpy().astype(int))

# Add another dimension to all for tensor model to take this data
# [256, 256, 3]    -->    [1, 256, 256, 3]
yhat_test = model.predict(np.expand_dims(resize/255,0))

print('****************************************************************************************************')
print('****************************************************************************************************')
print(yhat_test)
print('---- ID Num:         ', np.argmax(yhat_test)+1, ', Probability: ', yhat_test.max())
# print('---- 2nd Highest:    ', np.argmax(yhat_test), ', Probability: ', yhat_test.max())
# print('---- Checksum for Probability: ', np.sum(yhat_test))
print('****************************************************************************************************')
print('****************************************************************************************************')

# if yhat_test[10] > 0.8:
#     print(f'Predicted: Port Jackson')
# else:
#     print(f'Predicted: Not PJ')


##################################################################
## 11. Save the Model
# Bring in the load model dependancy --> can then load up the saved model
# from tensorflow.keras.models import load_model

# Save the model
# Saving it as a .h5 file serialises it onto something you can store as a disk
#  Similar to zipping a dataset --> serialisation file format
print('Saving Trained Model')
model.save(os.path.join('models','famClassModel.h5'))


#  Load the model: give the full directory name using os
print('Loading Trained Model')
new_model = load_model(os.path.join('models','famClassModel.h5'))

print('Classifying new image')
yhatnew_test = new_model.predict(np.expand_dims(resize/255, 0))
# if yhatnew[10] > 0.8:
#     print(f'Predicted: Port Jackson')
# else:
#     print(f'Predicted: Not PJ')

# position 9 should be highest as this represents port jackson shark
print('****************************************************************************************************')
print('****************************************************************************************************')
print('---- ID Num', np.argmax(yhatnew_test)+1, ', Probability: ', yhatnew_test.max())
# print('---- Checksum for Probability: ', np.sum(yhatnew_test))
print('****************************************************************************************************')
print('****************************************************************************************************')

# plot show at the end! 
# Halts the program to ensure figures stay viewable with program paused and not ended
plt.show()
