import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

class ModelManager:
    def __init__ (self, input_shape):
        self.input_shape = input_shape
        self.model = self._create_architecture()

    def _create_architecture(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation = "relu", input_shape= (self.input_shape,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation= "relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(8, activation= "relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation = "sigmoid")
        ])
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val):
        early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=5, verbose=1)
        print("---Training Starts---")
        history = self.model.fit(X_train, y_train, epochs = 10, batch_size = 2048, validation_data=(X_val, y_val), callbacks= [early_stop], verbose=1)
        return history