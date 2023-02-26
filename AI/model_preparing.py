import numpy as np
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from datetime import date


class MlModel:
    def __init__(self) -> None:
        self.image_height = 256
        self.image_width = 256
        self.image_channels = 3  # RGB
        self.batch_size = 8
        self.validation_split = 0.3
        self.epochs = 50
        self.should_evaluate = True
        self.should_plot = True

    def prepare_model(self, training_data, validation_data, class_names):
        num_classes = len(class_names)

        data_augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.experimental.preprocessing.RandomFlip("vertical",
                                                                      input_shape=(self.image_width,
                                                                                   self.image_height,
                                                                                   3)),
                tf.keras.layers.experimental.preprocessing.RandomRotation(0.4),
                tf.keras.layers.experimental.preprocessing.RandomZoom(0.3),
                tf.keras.layers.experimental.preprocessing.RandomTranslation(
                    height_factor=0.2, width_factor=0.2, fill_mode="wrap"),
                tf.keras.layers.experimental.preprocessing.RandomContrast(
                    factor=0.2)
            ]
        )

        model = tf.keras.Sequential([
            data_augmentation,
            tf.keras.layers.Resizing(
                self.image_height, self.image_width, interpolation="bilinear", crop_to_aspect_ratio=True),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(86, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(28, activation='relu'),
            tf.keras.layers.Dense(num_classes)
        ])

        # optimizer = tf.keras.optimizers.Adam(lr=0.001)
        optimizer = tf.keras.optimizers.experimental.SGD(lr=0.01)

        model.compile(
            optimizer=optimizer,
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        model.fit(
            training_data,
            validation_data=validation_data,
            epochs=self.epochs,
            shuffle=True,
            validation_split=self.validation_split
        )
        return model

    def __download_training_data(self,):
        dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
        data_dir = tf.keras.utils.get_file(origin=dataset_url,
                                           fname='flower_photos',
                                           untar=True)
        flowers_dir = pathlib.Path(data_dir)

        flower_train_df = tf.keras.utils.image_dataset_from_directory(
            flowers_dir,
            validation_split=self.validation_split,
            subset="training",
            image_size=(self.image_height, self.image_width),
            batch_size=self.batch_size,
            seed=1234
        )

        flower_val_df = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=self.validation_split,
            subset="validation",
            image_size=(self.image_height, self.image_width),
            batch_size=self.batch_size,
            seed=1234
        )
        self.class_names = flower_train_df.class_names
        return (flower_train_df, flower_val_df)

    def __configure_flower_dataset(self, train_df, val_ds):
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_df.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(
            1./255)

        normalized_train = train_ds.map(
            lambda x, y: (normalization_layer(x), y))
        normalized_valid = train_ds.map(
            lambda x, y: (normalization_layer(x), y))
        return (normalized_train, normalized_valid)

    def __create_flower_model(self, train_df, val_df):
        num_classes = 5
        data_augmentation = tf.keras.models.Sequential(
            [
                tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal",
                                                                      input_shape=(self.image_height,
                                                                                   self.image_width,
                                                                                   self.image_channels)),
                tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
                tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
            ]
        )
        model = Sequential([
            data_augmentation,
            tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
            tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(num_classes)
        ])
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(
                          from_logits=True),
                      metrics=['accuracy'])

        self.flower_model_history = model.fit(
            train_df,
            validation_data=val_df,
            epochs=self.epochs
        )

        return model

    def __create_plot(self):
        acc = self.flower_model_history.history['accuracy']
        val_acc = self.flower_model_history.history['val_accuracy']

        loss = self.flower_model_history.history['loss']
        val_loss = self.flower_model_history.history['val_loss']

        epochs_range = range(self.epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

    def __evaluate_flower_model(self, model):
        sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
        sunflower_path = tf.keras.utils.get_file(
            'Red_sunflower', origin=sunflower_url)

        img = tf.keras.preprocessing.image.load_img(
            sunflower_path, target_size=(self.image_height, self.image_width)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(self.class_names[np.argmax(score)], 100 * np.max(score))
        )

    def prepare_flower_model(self):
        train_df, val_df = self.__download_training_data()
        train_df, val_df = self.__configure_flower_dataset(train_df, val_df)
        model = self.__create_flower_model(train_df, val_df)
        if self.should_evaluate:
            self.__evaluate_flower_model(model)
        if self.should_plot:
            self.__create_plot()
