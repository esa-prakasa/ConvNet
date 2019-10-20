# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import os



epochSz = 3
dimSz = 300

# Pembuatan Arsitektur Deep Learning

classifier = Sequential()
classifier.add(Conv2D(64,(3,3),input_shape=(dimSz,dimSz,3),activation='relu'))
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
classifier.add(Dense(units =12,activation="softmax"))

classifier.compile(optimizer ='adam', loss = 'categorical_crossentropy', metrics =['accuracy'])
os.system('cls')  #>> for Windows
#os.system('clear')  #>> for Linux

classifier.summary()



print("Epoch and Dimension size are "+str(epochSz)+" and "+str(dimSz)+" repectively.")

expPath = 'C:\\Users\\INKOM06\\Documents\\GitHub\\ConvNet\\exp1020\\'
dataPath = 'C:\\Users\\INKOM06\\Pictures\\data12div01\\'

#os.chdir('/home/lgr0270013/esa/exp1014/')
modelFileNm = 'modnet'+str(dimSz)+'_'+str(epochSz)+'.json'

with open(expPath+modelFileNm, 'w') as f:  #>> Windows version
    f.write(classifier.to_json())


#os.chdir('/home/lgr0270013/0D_data12div/')

train_data_dir = dataPath+'train'  
valid_data_dir = dataPath+'valid'  
test_data_dir  = dataPath+'test' 




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

bestWgFileNm = 'best_weights12_'+str(dimSz)+'_'+str(epochSz)+'.hfd5'

checkpointer = ModelCheckpoint(filepath=expPath+bestWgFileNm, 
	monitor = 'val_acc',
	verbose=1, 
	save_best_only=True)


history = classifier.fit_generator(training_set,
	steps_per_epoch = 20,
	epochs = 3,
	callbacks=[checkpointer],
	validation_data = valid_set,
	validation_steps = 20)


#os.chdir('/home/lgr0270013/esa/exp1014/')

finalWgFileNm = 'final_weights12_'+str(dimSz)+'_'+str(epochSz)+'.h5'

classifier.save_weights(expPath+finalWgFileNm)
print("Saved model to disk")

