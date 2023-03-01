from .image_processing import ImagePreprocessing
from .model_preparing import MlModel
import tensorflow as tf
from pathlib import Path
import numpy as np
from .plant_rest_handler import PlantRestHandler
from difflib import SequenceMatcher
import time
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class AiHandler:
    def __init__(self) -> None:
        self.enable_fallback = True
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
        timestr = time.strftime("%Y%m%d-%H%M%S")
        self.save_model(Path(Path.cwd() / "models" / f"model_{timestr}.h5"))

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)

    def compare_strings(self, target_string, string_list):
        max_score = 0
        best_match = ""
        score = 0
        for string in string_list:
            score = SequenceMatcher(None, target_string, string).ratio()
            if score > max_score:
                print(f"score is {score}")
                max_score = score
                best_match = string
        
        if score < 0.6:
            best_match = target_string

        return best_match

    def predict_image(self, server_state, image_path):
        image_preprocessor = ImagePreprocessing()
        input_data = image_preprocessor.load_image(image_path)
        if not self.model:
            self.load_model(server_state.model_path)
        if not self.class_names:
            self.class_names = server_state.supported_plants

        prediction = self.model.predict(input_data)
        label = self.class_names[np.argmax(prediction)]
        if self.enable_fallback:
            _, fallback_plant_name = PlantRestHandler.identify_plant(image_path)
            label = self.compare_strings(fallback_plant_name, server_state.supported_plants)

        if server_state.plants_info.get(label):
            response = server_state.plants_info[label]
        else:
            response = {"Plant name": label}
        return response