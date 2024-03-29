#install dependencies
#import packages in the project
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#initializing a CNN
classifier = Sequential() 

#add convolutional layer
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3), activation= 'relu')) 

#adding pooling layer
classifier.add(MaxPooling2D(pool_size=(2,2)))

#adding flattening layer
classifier.add(Flatten())

#Adding fully connected layer
#for hidden layer and for output layer
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units= 1, activation='sigmoid')) 
#sigmoid activation  function is used t return the possiblity of ccurance of dogs and cats

#compiling the CNN
classifier.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
                            'dataset/training_set',
                            target_size=(64,64),
                            batch_size=32,
                            class_mode='binary')

test_set = test_datagen.flow_from_directory(
                            'dataset/test_set',
                            target_size=(64,64),
                            batch_size=32,
                            class_mode='binary')


classifier.fit_generator(training_set,
                            steps_per_epoch= int(8000),
                            epochs=25,
                            validation_data=test_set,
                            validation_steps= int(2000))


