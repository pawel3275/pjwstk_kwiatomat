import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class MlModel:
    def __init__(self) -> None:
        self.image_height = 256
        self.image_width = 256
        self.image_channels = 3  # RGB
        self.batch_size = 8
        self.validation_split = 0.3
        self.epochs = 100

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
            tf.keras.layers.Conv2D(186, 3, activation='relu'),
            tf.keras.layers.Conv2D(128, 3, activation='relu'),
            tf.keras.layers.Conv2D(96, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(156, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(192, activation='relu'),
            tf.keras.layers.Dense(132, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(86, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(48, activation='relu'),
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
