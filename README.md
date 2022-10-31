# Inteligencia-Artificial Version1


### Entrenar
```python
import sys
import os
#from tensorflow.python.keras import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#from tensorflow.python.keras import optimizers #esta linea es la original
#from tensorflow.python.keras.optimizers import Adam #linea original
from tensorflow.keras.optimizers import Adam #codigo copiado que funciona


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.keras.layers import  Convolution2D, MaxPooling2D
from tensorflow.keras import backend as K

#codigo experimental
import tensorflow as tf
opta = tf.keras.optimizers.Adam(lr=0.0004)

#codigo2
#from keras.optimizers import adam_v2
#opt = adam_v2.Adam(learning_rate=lr, decay=lr/epochs)

#import tensorflow as tf
#adam = optimizers.adam_v2.Adam(lr=0.01) #codigo copiado

#codigo 3
#from tensorflow.keras.optimizers import RMSprop
#opta = RMSprop(lr=0.0001, decay=1e-6)

K.clear_session()



data_entrenamiento = './data/entrenamiento'
data_validacion = './data/validacion'

"""
Parameters
"""
epocas=20
longitud, altura = 150, 150
batch_size = 32
pasos = 1000
validation_steps = 300
filtrosConv1 = 32
filtrosConv2 = 64
tamano_filtro1 = (3, 3)
tamano_filtro2 = (2, 2)
tamano_pool = (2, 2)
clases = 3
lr = 0.0004


##Preparamos nuestras imagenes

entrenamiento_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

entrenamiento_generador = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')

validacion_generador = test_datagen.flow_from_directory(
    data_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')

cnn = Sequential()
cnn.add(Convolution2D(filtrosConv1, tamano_filtro1, padding ="same", input_shape=(longitud, altura, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding ="same"))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Flatten())
cnn.add(Dense(256, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(clases, activation='softmax'))

cnn.compile(loss='categorical_crossentropy',
            optimizer=opta,
            metrics=['accuracy'])

cnn.fit_generator(
    entrenamiento_generador,
    steps_per_epoch=pasos,
    epochs=epocas,
    validation_data=validacion_generador,
    validation_steps=validation_steps)

target_dir = './modelo/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
cnn.save('./modelo/modelo.h5')
cnn.save_weights('./modelo/pesos.h5')
```
### Predecir
```python
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

longitud, altura = 150, 150
modelo = './modelo/modelo.h5'
pesos_modelo = './modelo/pesos.h5'
cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)

def predict(file):
  x = load_img(file, target_size=(longitud, altura))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  arreglo = cnn.predict(x)
  resultado = arreglo[0]
  respuesta = np.argmax(resultado)
  if respuesta == 0:
    print("pred: Perro")
  if respuesta == 1:
    print("pred: Gato")
  if respuesta == 2:
    print("pred: Gorila")

  return respuesta

predict('dog.4226.jpg')
```
