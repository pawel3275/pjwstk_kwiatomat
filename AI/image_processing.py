import cv2
from PIL import Image
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras.utils as keras_image
import numpy as np
import cv2
import matplotlib.pyplot as plt

class ImagePreprocessing:
    def __init__(self) -> None:        
        self.dataset_path = "dataset"
        self.desired_image_size = (256, 256)
        self.train_dataset = None
        self.validation_dataset = None

    def resize_image(self, path):
        if os.path.isfile(path):
            im = Image.open(path).resize(self.desired_image_size, Image.ANTIALIAS)
            parent_dir = os.path.dirname(path)
            img_name = os.path.basename(path).split('.')[0]
            im.save(os.path.join(parent_dir, img_name + '_r.jpeg'), 'JPEG', quality=90)
            # Remove original file:
            os.remove(path)

    def _resize_all(self):
        for subdir , _ , fileList in os.walk(self.dataset_path):
            for f in fileList:
                try:
                    full_path = os.path.join(subdir,f)
                    self.resize_image(full_path)
                except Exception as e:
                    os.remove(full_path)
                    print(f'Unable to resize {full_path}. Deleting.')

    def preprocess_data(self):
        #self._resize_all()
        return self._prepare_data_for_model()

    def _prepare_data_for_model(self):
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
        print(f"Class names are: {self.class_names}")

        normalization_layer = tf.keras.layers.Rescaling(1./255)
        self.train_dataset = training_data.map(lambda x, y: (normalization_layer(x), y))
        self.validation_dataset = validation_data.map(lambda x, y: (normalization_layer(x), y))

        print("Data Augmentation Completed")
        return (self.train_dataset, self.validation_dataset)

    def rescale_pixels_of_image(self, path):
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
        image = cv2.imread(image_path)
        cv2.imshow(image_path, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
