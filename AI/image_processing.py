import cv2
import keras.utils as keras_image
import numpy as np
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class ImagePreprocessing:
    """
    A class for preprocessing images for machine learning.
    """
    def __init__(self) -> None:        
        self.dataset_path = "dataset"
        self.desired_image_size = (256, 256)
        self.train_dataset = None
        self.validation_dataset = None

    def preprocess_data(self):
        """
        Preprocess image data and prepare for use in a model.

        Returns:
            tuple: A tuple containing the processed training dataset and validation dataset.
        """
        return self._prepare_data_for_model()

    def _prepare_data_for_model(self):
        """
        Private method to preprocess data for use in a model.

        Returns:
            tuple: A tuple containing the processed training dataset and validation dataset.
        """
        training_data = tf.keras.preprocessing.image_dataset_from_directory(
            self.dataset_path,
            validation_split=0.2,
            subset="training",
            crop_to_aspect_ratio=True,
            image_size=self.desired_image_size,
            batch_size=8,
            seed=123,
            shuffle=True
        )

        validation_data = tf.keras.preprocessing.image_dataset_from_directory(
            self.dataset_path,
            validation_split=0.2,
            subset="validation",
            crop_to_aspect_ratio=True,
            image_size=self.desired_image_size,
            batch_size=8,
            seed=123,
            shuffle=True
        )

        self.class_names = training_data.class_names
        np.array(self.class_names).tofile("model_labels.txt", ",")

        normalization_layer = tf.keras.layers.Rescaling(1./255)
        self.train_dataset = training_data.map(lambda x, y: (normalization_layer(x), y))
        self.validation_dataset = validation_data.map(lambda x, y: (normalization_layer(x), y))

        print("Data Augmentation Completed")
        return (self.train_dataset, self.validation_dataset)

    def load_image(self, path):
        """
        Load an image from a file path and preprocess it for use in a model.

        Args:
        path (str): Path to the image file.

        Returns:
            numpy.ndarray: A preprocessed numpy array representing the image.
        """
        img = keras_image.load_img(
            path, 
            target_size=self.desired_image_size
        )
        img_tensor = keras_image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.
        return img_tensor

    @staticmethod
    def show_image(image_path):
        """
        Display an image from a file path.

        Args:
            image_path (str): Path to the image file.
        """
        image = cv2.imread(image_path)
        cv2.imshow(image_path, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
