from .image_processing import ImagePreprocessing
from .model_preparing import MlModel
import tensorflow as tf
from pathlib import Path
import numpy as np

class AiHandler:
    def __init__(self) -> None:
        
        self.model_preprocessor = MlModel()
        self.model_path = None
        self.validation_data = None
        self.training_data = None
        self.class_names = None
        self.model = None

    def preprocess_images(self):
        self.image_preprocessor = ImagePreprocessing()
        self.training_data, self.validation_data = self.image_preprocessor.preprocess_data()
        self.class_names = self.image_preprocessor.class_names

    def train_model(self):
        model_instance = MlModel()
        self.model = model_instance.prepare_model(
            self.training_data,
            self.validation_data,
            self.class_names
        )
        self.save_model(Path(Path.cwd() / "models" / "testing_model11.h5"))

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)

    def predict_image(self, server_state, image_path):
        image_path = "D:/scratch/inzynierka/test_r_r.jpeg"
        image_preprocessor = ImagePreprocessing()
        input_data = image_preprocessor.rescale_pixels_of_image(image_path)
        if not self.model:
            self.load_model(server_state.model_path)
        if not self.class_names:
            self.class_names = server_state.supported_plants
        prediction = self.model.predict(input_data)
        print(f"PREDICTION IS {prediction}")
        label = self.class_names[np.argmax(prediction)]
        print(f"LABEL IS {label}")
        print(f"json output will be: \n\n {server_state.plants_info[label]}")
        return server_state.plants_info[label]