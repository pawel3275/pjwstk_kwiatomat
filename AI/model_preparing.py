import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.metrics import confusion_matrix
from transformers import BeitFeatureExtractor, BeitForImageClassification
from vit_keras import vit

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


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
        self.enable_vim_l32 = True

    def get_data_augmentation(self):
        """
        Returns a Keras sequential model for data augmentation of images.
        The model applies various random transformations to the input images, including
        vertical flipping, rotation, zooming, translation, and contrast adjustment.

        Returns:
        - A Keras sequential model consisting of the following layers:
            - RandomFlip layer that flips the input images along the vertical axis randomly
            - RandomRotation layer that applies random rotations to the input images up to 40%
            - RandomZoom layer that randomly zooms in or out of the input images up to 30%
            - RandomTranslation layer that applies random translations to the input images up to 20%
            - RandomContrast layer that applies random contrast adjustments to the input images up to 20%
        """
        return tf.keras.Sequential(
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

    def get_vit_l32_model(self, num_classes):
        """
        Creates a Keras sequential model that uses a pre-trained ViT-L32 model.

        Args:
            num_classes (int): The number of classes in the classification task.

        Returns:
            A Keras sequential model that can be trained and evaluated for image classification.
            The model uses a pre-trained ViT-L32 model as a feature extractor, followed by
            a flatten layer, two dense layers, and a softmax activation layer.
        """
        # define a Keras model that uses a pre-trained ViT model
        pretrained_model = vit.vit_l32(
            image_size=256,
            pretrained=True,
            include_top=False,
            pretrained_top=False)

        model = tf.keras.Sequential([
            tf.keras.layers.Input(
                shape=(self.image_height, self.image_width, self.image_channels)),
            self.get_data_augmentation(),
            pretrained_model,
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

        return model

    @staticmethod
    def use_beit_model(image_path):
        """
        Uses the Beit transformer model for image classification to predict the class of the given image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            str: The predicted class label of the input image.
        """
        image = Image.open(image_path)
        feature_extractor = BeitFeatureExtractor.from_pretrained(
            'microsoft/beit-base-patch16-224-pt22k-ft22k')
        model = BeitForImageClassification.from_pretrained(
            'microsoft/beit-base-patch16-224-pt22k-ft22k')
        inputs = feature_extractor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        print("Predicted class:", model.config.id2label[predicted_class_idx])
        return model.config.id2label[predicted_class_idx]

    def get_convolutional_model(self, num_classes):
        """Creates and returns a convolutional neural network (CNN) model for image classification.

        Args:
            num_classes (int): The number of classes (i.e., unique labels) in the dataset.

        Returns:
            tensorflow.python.keras.engine.training.Model: A compiled CNN model for image classification.

        """
        model = tf.keras.Sequential([
            self.get_data_augmentation(),
            tf.keras.layers.Resizing(
                self.image_height, self.image_width, interpolation="bilinear", crop_to_aspect_ratio=True),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(
                self.image_height, self.image_width, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(36, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        return model

    def prepare_plots(self, history, timestamp):
        """Plots and saves training and validation accuracy and loss curves.

        Args:
            history (tensorflow.python.keras.callbacks.History): The history object returned by the fit() method.
            timestamp (str): A timestamp string used to generate the filename of the saved plot.

        Returns:
            None.

        """
        loss_file = Path(f"models/model_loss_{timestamp}.png")
        acc_file = Path(f"models/model_acc_{timestamp}.png")

        # plot training and validation accuracy values
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(acc_file)

        # plot training and validation loss values
        plt.clf()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(loss_file)

    def prepare_confusion_matrix(self, model, validation_data, timestamp, class_names):
        """Creates and saves a confusion matrix plot.

        Args:
            model (tensorflow.python.keras.engine.training.Model): The trained model to evaluate.
            validation_data (tensorflow.python.keras.utils.Sequence): The validation data generator to use.
            timestamp (str): A timestamp string used to generate the filename of the saved plot.
            class_names (list): A list of class names corresponding to the labels in the dataset.

        Returns:
            None.

        """
        # Get the true labels and predicted labels
        y_true = np.concatenate([y for _, y in validation_data], axis=0)
        y_pred = np.argmax(model.predict(validation_data), axis=1)

        # Create the confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Plot the confusion matrix using seaborn
        conf_file = Path(f"models/confusion_matrix_{timestamp}.png")
        plt.figure(figsize=(20, 16))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.savefig(conf_file)

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
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Training of model started at {timestamp}")
        num_classes = len(class_names)

        if self.enable_vim_l32:
            model = self.get_vit_l32_model(num_classes)
        else:
            model = self.get_convolutional_model(num_classes)

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
        print(f"Training of model ended at {timestamp}")
        self.prepare_plots(history, timestamp)
        self.prepare_confusion_matrix(
            model, validation_data, timestamp, class_names)

        return model
