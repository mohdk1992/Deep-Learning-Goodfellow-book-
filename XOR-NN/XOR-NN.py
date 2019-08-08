# Neural Network for solving XOR problem

from tensorflow import keras
import numpy as np

# XOR Data

x_data = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

y_data = [
    [0],
    [1],
    [1],
    [0]
]

x_data = np.array(x_data)
y_data = np.array(y_data)


# Keras model
model = keras.Sequential()

model.add(keras.layers.Dense(2, activation = "sigmoid", input_shape = (2,)))
model.add(keras.layers.Dense(1, activation = "sigmoid"))

optimizer = keras.optimizers.Adam(lr = 0.01)

model.compile(optimizer, loss = "binary_crossentropy", metrics=["accuracy"])
model.summary()

model.fit(x_data, y_data, epochs= 1000)

predict = model.predict((x_data))
print(predict)



