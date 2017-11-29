#Pkg 
import numpy as np
np.random.seed(123)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import optimizers
from keras import initializers
import os
import numpy as np
import random
from keras.constraints import Constraint
from keras.callbacks import Callback
import keras
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.models import model_from_json

keras.backend.set_image_data_format('channels_first')


#Parameters
classes = 4
n_epochs = 12
iterations_per_epoch = 600
batch = 32

sgd_learning_rate = 0.000001
sgd_decay = 0.0005
sgd_momentum = 0.9

image_row = 227
image_col = 227


#Function to normalise and prepare univ layer kernel
def normalise(w):
	j = int(w.shape[0]/2)
	for i in range(w.shape[-1]):
		w[j,j,:,i]= 0
		wsum = w[:,:,:,i].sum()
		w[:,:,:,i]/=wsum
		w[j,j,:,i]=-1
	return w

class Normalise(Callback):	
	def on_batch_end(self, batch, logs=None):	
		total_w = self.model.layers[0].get_weights()
		w = np.array(total_w[0])
		bias = np.array(total_w[1])
		w = normalise(w)
		self.model.layers[0].set_weights([w, bias])

#Loading Data
train_datagen = ImageDataGenerator(data_format = 'channels_first',rescale=1./255)

test_datagen = ImageDataGenerator(data_format = 'channels_first',rescale=1./255)

train_generator = train_datagen.flow_from_directory(
		'./CNN_Data/Train',
		target_size=(image_row, image_col),
		color_mode = "grayscale",
		batch_size=batch,
		class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
		'./CNN_Data/Validation',
		target_size=(image_row, image_col),
		color_mode="grayscale",
		batch_size=batch,
		class_mode='categorical')

#convRes weight init
w = np.random.rand(5,5,1,12)
wgt = normalise(w)
bias = np.zeros(12)


#Architecture
model = Sequential()

model.add(Convolution2D(12, (5, 5), input_shape=(1,image_row,image_col))) 			#convRes
model.layers[0].set_weights([wgt, bias])

model.add(Convolution2D(64, (7, 7), activation='relu', strides=(2,2)))           	#conv1

model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))                   			#Max Pooling

model.add(Convolution2D(48, (3, 3), activation='relu', strides = (1,1)))		  	#conv2

model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

model.add(Flatten())																#FC1
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(4096, activation='relu'))											#FC2
model.add(Dropout(0.5))

model.add(Dense(classes, activation='softmax'))


#Model
sgd = optimizers.SGD(lr = sgd_learning_rate, decay = sgd_decay, momentum = sgd_momentum)
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])

#Fitting
norm_callback = Normalise()

model.fit_generator(
		train_generator,
		callbacks = [norm_callback],
		steps_per_epoch=iterations_per_epoch,
		epochs = n_epochs,
		validation_data = validation_generator,
		validation_steps = 240,
		verbose = 1)

#Evaluation
score = model.evaluate_generator(validation_generator, 240)
print('Loss: ', score[0], 'Accuracy:', score[1])

#Saving
model_json = model.to_json()
with open("univ_model.json","w") as json_file:
	json_file.write(model_json)

model.save_weights("univ_model_wgts.h5")


