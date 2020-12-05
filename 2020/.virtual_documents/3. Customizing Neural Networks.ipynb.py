from sklearn.datasets import make_regression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


features, targets = make_regression(n_samples=500, 
                                    n_features=1000, 
                                    n_targets=10)


model = Sequential()


layer = Dense(500, input_dim=1000, 
                kernel_initializer='he_uniform', 
                activation='relu')
model.add(layer)


layer = Dense(250, activation='relu')
model.add(layer)


layer = Dense(125, activation='relu')
model.add(layer)


layer = Dense(50, activation='relu')
model.add(layer)


layer = Dense(10)
model.add(layer)


model.compile(loss="mean_squared_error", 
              optimizer='adam')


predictions = model.predict(features)
predictions.shape


from keras import backend as K
from keras.callbacks import TensorBoard

# Define Tensorboard as a Keras callback
tensorboard = TensorBoard(
  log_dir='./new_logs',
  histogram_freq=1,
  write_images=True
)
keras_callbacks = [
  tensorboard
]

# Fit data to model
model.fit(features, targets, epochs=0,
          callbacks=keras_callbacks)


# tensorboard --logdir=./logs



