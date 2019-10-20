# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import os

# Pembuatan Arsitektur Deep Learning

dimSz = 100

classifier = Sequential()
classifier.add(Conv2D(128,(3,3),input_shape=(dimSz,dimSz,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Conv2D(64,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Conv2D(32,(3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Conv2D(16,(3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(MaxPooling2D(pool_size=(4,4)))

classifier.add(Flatten())
classifier.add(Dense(units =1024,activation="relu"))
classifier.add(Dense(units =256,activation="relu"))
classifier.add(Dense(units =128,activation="relu"))
classifier.add(Dense(units =3,activation="softmax"))

classifier.compile(optimizer ='adam', loss = 'categorical_crossentropy', metrics =['accuracy'])
os.system('cls')  #>> for Windows
#os.system('clear')  #>> for Linux

classifier.summary()

print("Demension size is "+str(dimSz))


#os.chdir('/home/lgr0270013/esa/exp1014/')

with open("C:\\Users\\INKOM06\\kerasCodes\\exp1018_loc\\modnet400_30.json", 'w') as f:  #>> Windows version
    f.write(classifier.to_json())


#os.chdir('/home/lgr0270013/0D_data12div/')

train_data_dir = 'C:\\Users\\INKOM06\\Pictures\\data2\\train'  
valid_data_dir = 'C:\\Users\\INKOM06\\Pictures\\data2\\valid'  
test_data_dir = 'C:\\Users\\INKOM06\\Pictures\\data2\\test' 




#Data Augmentation
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
	shear_range = 0.02,
	rotation_range = 30,
	horizontal_flip = False)
# Setting Lokasi Dataset
valid_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(train_data_dir,
	target_size = (dimSz, dimSz),
	batch_size = 32,
	class_mode = 'categorical')
valid_set = valid_datagen.flow_from_directory(valid_data_dir,
	target_size = (dimSz, dimSz),
	batch_size = 32,
	class_mode = 'categorical')


from keras.callbacks import ModelCheckpoint


checkpointer = ModelCheckpoint(filepath="C:\\Users\\INKOM06\\kerasCodes\\exp1018_loc\\best_weights_wood12C400_30.hdf5", 
	monitor = 'val_acc',
	verbose=1, 
	save_best_only=True)


history = classifier.fit_generator(training_set,
	steps_per_epoch = 20,
	epochs = 10,
	callbacks=[checkpointer],
	validation_data = valid_set,
	validation_steps = 20)


#os.chdir('/home/lgr0270013/esa/exp1014/')

classifier.save_weights("C:\\Users\\INKOM06\\kerasCodes\\exp1018_loc\\wood12C400_30.h5")
print("Saved model to disk")

