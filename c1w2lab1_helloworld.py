import tensorflow as tf

fmist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = fmist.load_data()

training_images = training_images / 255.0
test_images = test_images / 255.0


# define callback
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('loss') < 0.4:
            print("\nLoss is lower than 0.4 so cancelling taining!")
            self.model.stop_training = True


callbacks = myCallback()

# craete a model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])

# fit the model
model.fit(training_images, training_labels, epochs=10, callbacks=[callbacks])

# evaluate the model
model.evaluate(test_images, test_labels)
