import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#70k images, 60k for training, 10k for testing
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#print(train_labels[0])  
#print(train_images[0])
#plt.imshow(train_images[32456], cmap='gray', vmin=0, vmax=255)
#plt.show()


#NEURAL_NET_STRUCTURE#
model = keras.Sequential([

    #input layer, each neuron will have 1 pixel of image, 28x28 matrix shape flattened into 784x1
    #flatten to simplify the structure of the neural net
    keras.layers.Flatten(input_shape = (28, 28)), 

    #hidden layer, 128 neurons, relu returns value or 0
    #softmax loss was not so good
    keras.layers.Dense(units=128, activation=tf.nn.relu),

    #output layer, 0-9, return max, dense is each neuron is connected to each neuron
    keras.layers.Dense(units=10, activation=tf.nn.softmax)

])

#compile model, optimizer changes the numbers to minimize loss (weights, biases)
model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


#TRAINING_THE_NETWORK#
#train model, epoch is a pass of the training data and optimization
model.fit(train_images, train_labels, epochs = 5)

#test model, gives overall loss on the test images
test_loss = model.evaluate(test_images, test_labels)


#PREDICTION#
#plt.imshow(test_images[0], cmap='gray', vmin=0, vmax=255)
#plt.show()
print(test_labels[0])  

#make predictions
predictions = model.predict(test_images)
print(predictions[0])
print(list(predictions[0]).index(max(predictions[0])))


print("bruh")

