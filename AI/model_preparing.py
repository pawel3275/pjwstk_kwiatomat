import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import datetime
from pathlib import Path

import matplotlib.pyplot as plt


class MlModel:
    """
    A class representing a machine learning model for image classification.
    """
    def __init__(self) -> None:
        self.image_height = 256
        self.image_width = 256
        self.image_channels = 3  # RGB
        self.batch_size = 8
        self.validation_split = 0.3
        self.epochs = 70

    def prepare_model(self, training_data, validation_data, class_names):
        """
        Trains and returns a machine learning model using the provided training and validation data and class names.

        Args:
        - training_data (tf.data.Dataset): The training data for the model.
        - validation_data (tf.data.Dataset): The validation data for the model.
        - class_names (list): A list of class names for the model.

        Returns:
        - model (tf.keras.Sequential): A machine learning model trained on the provided data and class names.
        """
        num_classes = len(class_names)

        data_augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.experimental.preprocessing.RandomFlip(
                    "vertical",
                    input_shape=(
                        self.image_width,
                        self.image_height,
                        3
                    )
                ),
                tf.keras.layers.experimental.preprocessing.RandomRotation(0.4),
                tf.keras.layers.experimental.preprocessing.RandomZoom(0.3),
                tf.keras.layers.experimental.preprocessing.RandomTranslation(
                    height_factor=0.2, 
                    width_factor=0.2, 
                    fill_mode="wrap"
                ),
                tf.keras.layers.experimental.preprocessing.RandomContrast(
                    factor=0.2)
            ]
        )

        model = tf.keras.Sequential([
            data_augmentation,
            tf.keras.layers.Resizing(
                self.image_height, self.image_width, interpolation="bilinear", crop_to_aspect_ratio=True),
                tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(self.image_height, self.image_width, 3)),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(num_classes, activation='softmax')
            ])


        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=10000,
            decay_rate=0.9)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        history = model.fit(
            training_data,
            validation_data=validation_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            validation_split=self.validation_split,
            callbacks=[early_stopping]
        )

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_file = Path(f"models/model_{timestamp}.png")
        # plot training and validation accuracy values
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig('accuracy.png')

        # plot training and validation loss values
        plt.clf()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(chart_file)

        return model
