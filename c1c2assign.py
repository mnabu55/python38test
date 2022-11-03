import tensorflow as tf

fmist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = fmist.load_data()

x_train = x_train / 255.0

data_shape = x_train.shape

print(f"There are {data_shape[0]} examples with shape ({data_shape[1]}, {data_shape[2]})")

# define callback
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, log={}):
        if log.get('loss') < 0.4:
            print("\nLoss is lower than 0.4, so cancelling training")
            self.model.stop_training = True


def train_mnist(x_train, y_train):
    callbacks = myCallback()

    # create a model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # history = model.fit(x_train, y_train, epochs=5, callbacks=[callbacks])
    history = model.fit(x_train, y_train, epochs=5)

    return history


history = train_mnist(x_train, y_train)

print(history.history)

import matplotlib.pyplot as plt


#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc = history.history['accuracy']
#val_acc = history.history['val_accuracy']
loss = history.history['loss']
#val_loss = history.history['val_loss']

epochs = range(len(acc))    # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc)
#plt.plot(epochs, val_acc)
plt.title('Training accuracy')
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss)
# plt.plot  ( epochs, val_loss )
plt.title ('Training loss'   )

plt.show()
