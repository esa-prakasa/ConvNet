# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_json

import os

rootPath = 'C:\\Users\\INKOM06\\Pictures\\data12div01\\' 

train_data_dir = rootPath+'train'  
valid_data_dir = rootPath+'valid'  
test_data_dir  = rootPath+'test' 
model_dir      = rootPath+'exp1020mowg' 

target_names = [item for item in os.listdir(train_data_dir) if os.path.isdir(os.path.join(train_data_dir, item))]


# load json and create model
json_file = open(model_dir+'\\'+'modnet400_30.json', 'r')  #>>> Windows version



loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
#model.load_weights('/content/drive/My Drive/data12_C/woodcls_dz600_ep30_withDropOut.h5')
model.load_weights(model_dir+'\\'+'best_weights_wood12C400_30.hdf5') #>>> Windows version
os.system('cls')
target_names = sorted(target_names)


print("Loaded model from disk")
print(target_names)
dimSz = 100


## ============================================== >>>>>>

import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
className = target_names
print(className)


rootDir =test_data_dir

idxFig = 0;

noTestedImages = 9


for idx in range(3):
  print('---------------',className[idx],'-------------------')
  correctAns = 0
  for idxSp in range(noTestedImages):
    #fullPath = rootDir+className[idx]+'/'+className[idx]+str(idxSp+1)+'.jpg'
    #print(fullPath)

    fullPath = rootDir+'\\'+className[idx]+'\\'; 
    arr = os.listdir(fullPath)
    #print(str(idx)+' '+str(idxSp)+' ')
    fullPathToRead = fullPath+arr[idxSp]
        
    test_image = image.load_img(fullPathToRead, target_size =(dimSz, dimSz))
    pics = test_image
    
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    #result = classifier.predict(test_image)
    result = model.predict(test_image)
    classIdx = np.argmax(result)
    #print(str(idxFig)+"  "+str(classIdx))
    #print("The image of "+className[idx]+str(idxSp+1)+" is identified as",className[classIdx], "  ",idxFig)  

    idxFig = idxFig + 1;    
    plt.subplot(12,15,idxFig)
    plt.imshow(pics)
    plt.title(className[idx]+":"+str(idxSp+1)+" --> "+className[classIdx])
    plt.subplots_adjust(wspace=0.5, hspace=0.3)
    print(className[idx]+" -> "+className[classIdx])

    if (className[idx] == className[classIdx]):
      correctAns = correctAns + 1
    accuracy = (correctAns/noTestedImages)*100
  
  print("%d: Accuracy of %s : %8.2f" % (idx, className[idx],accuracy))

