# Importing the Keras libraries and packages
#
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import os


# Create a simple ConvNet mode.
# Pembuatan Arsitektur Deep Learning
classifier = Sequential()
classifier.add(Conv2D(32,3,3,input_shape=(64,64,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())
classifier.add(Dense(output_dim =128,activation='relu'))
classifier.add(Dense(output_dim =1,activation='sigmoid'))
classifier.compile(optimizer ='adam', loss = 'binary_crossentropy', metrics =['accuracy'])
os.system('clear') 




#Data Augmentation. This part is used to create augemented images for training and validation purposes.
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
	shear_range = 0.02,
	rotation_range = 5,
	horizontal_flip = False)

# Setting Lokasi Dataset
valid_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/train',
	target_size = (64, 64),
	batch_size = 32,
	class_mode = 'binary')
valid_set = valid_datagen.flow_from_directory('dataset/valid',
	target_size = (64, 64),
	batch_size = 32,
	class_mode = 'binary')

classifier.fit_generator(
	training_set,
	steps_per_epoch = 50,
	epochs = 3,
	validation_data = valid_set,
	validation_steps = 20)


# Part 3 - Making new predictions
import numpy as np
from keras.preprocessing import image

test_image = image.load_img('dataset/test/matoa1.jpg', target_size =(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
	prediction = 'meranti'
else:
	prediction ='matoa'
print("Matoa1 >>> ",prediction,"  Class: ",result[0][0])

test_image = image.load_img('dataset/test/matoa2.jpg', target_size =(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
	prediction = 'meranti'
else:
	prediction ='matoa'
print("Matoa2 >>> ",prediction,"  Class: ",result[0][0])

test_image = image.load_img('dataset/test/matoa3.jpg', target_size =(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
	prediction = 'meranti'
else:
	prediction ='matoa'
print("Matoa3 >>> ",prediction,"  Class: ",result[0][0])

test_image = image.load_img('dataset/test/matoa4.jpg', target_size =(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
	prediction = 'meranti'
else:
	prediction ='matoa'
print("Matoa4 >>> ",prediction,"  Class: ",result[0][0])


test_image = image.load_img('dataset/test/matoa5.jpg', target_size =(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
	prediction = 'meranti'
else:
	prediction ='matoa'
print("Matoa5 >>> ",prediction,"  Class: ",result[0][0])







test_image = image.load_img('dataset/test/meranti1.jpg', target_size =(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
#training_set.class_indices
if result[0][0] == 1:
	prediction = 'meranti'
else:
	prediction ='matoa'
print("Meranti1 >>> ",prediction,"  Class: ",result[0][0])

test_image = image.load_img('dataset/test/meranti2.jpg', target_size =(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
#training_set.class_indices
if result[0][0] == 1:
	prediction = 'meranti'
else:
	prediction ='matoa'
print("Meranti2 >>> ",prediction,"  Class: ",result[0][0])

test_image = image.load_img('dataset/test/meranti3.jpg', target_size =(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
#training_set.class_indices
if result[0][0] == 1:
	prediction = 'meranti'
else:
	prediction ='matoa'
print("Meranti3 >>> ",prediction,"  Class: ",result[0][0])

test_image = image.load_img('dataset/test/meranti4.jpg', target_size =(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
#training_set.class_indices
if result[0][0] == 1:
	prediction = 'meranti'
else:
	prediction ='matoa'
print("Meranti4 >>> ",prediction,"  Class: ",result[0][0])

test_image = image.load_img('dataset/test/meranti5.jpg', target_size =(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
#training_set.class_indices
if result[0][0] == 1:
	prediction = 'meranti'
else:
	prediction ='matoa'
print("Meranti5 >>> ",prediction,"  Class: ",result[0][0])

