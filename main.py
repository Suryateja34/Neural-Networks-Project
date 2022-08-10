#This is the OCR Project on neural networks using Tensor flow
import os
import cv2 #CV2 is used to load process images
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

mnist = tf.keras.datasets.mnist

#X data is the picture and y data is the label of the data i.e., the number present in the picture
(x_train, y_train), (x_test,y_test) = mnist.load_data()

# #the next step is we should scale the data so that the pixel values which lie between 0 - 255 is converted to values between 0 - 1
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape = (28,28))) #Flatten converts a 28 * 28 pixels grid into a 786 pixels lenght line
model.add(tf.keras.layers.Dense(128, activation = "relu"))
model.add(tf.keras.layers.Dense(128, activation = "relu"))
model.add(tf.keras.layers.Dense(10, activation = "softmax"))

model.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 7)

model.save('handwritten.model')

model = tf.keras.models.load_model('handwritten.model')

loss, accuracy = model.evaluate(x_test,y_test)

print(loss)
print(accuracy)
image_number = 1
while os.path.isfile(f"Digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"Digits/digit{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()

    except:
        print("Error")
    finally:
        image_number += 1





