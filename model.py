import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPooling2D , Flatten , Dropout


(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train.shape , y_train.shape , X_test.shape , y_test.shape

X_train = X_train.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255

X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)


y_train = keras.utils.to_categorical(y_train)

y_test = keras.utils.to_categorical(y_test)


model = Sequential()

model.add(Conv2D(32, (3,3), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dropout(0.25))

model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


from keras.callbacks import ModelCheckpoint , EarlyStopping

es = EarlyStopping(monitor='val_accuracy', min_delta =  0.01 , verbose=1, patience=4)

mc = ModelCheckpoint('best_model.h5.keras', monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False)


cb = [es, mc]


his = model.fit(X_train, y_train, epochs=50,validation_split=0.3 ,callbacks=cb) 


model_S = keras.models.load_model('best_model.h5.keras')

score = model_S.evaluate(X_test, y_test)

print('the model accuracy is :', score[1])