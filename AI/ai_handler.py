from .image_processing import ImagePreprocessing
from .model_preparing import MlModel
import tensorflow as tf
from pathlib import Path
import numpy as np
from .plant_rest_handler import PlantRestHandler
from difflib import SequenceMatcher

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
        self.save_model(Path(Path.cwd() / "models" / "testing_model11.h5"))

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)

    def compare_strings(self, target_string, string_list):
        max_score = 0
        best_match = ""
        for string in string_list:
            score = SequenceMatcher(None, target_string, string).ratio()
            if score > max_score:
                max_score = score
                best_match = string
        print(f"BEST MATCH IS {best_match}")
        return best_match

    def predict_image(self, server_state, image_path):
        image_path = "D:/scratch/inzynierka/test_r_r.jpeg"
        image_preprocessor = ImagePreprocessing()
        input_data = image_preprocessor.rescale_pixels_of_image(image_path)
        if not self.model:
            self.load_model(server_state.model_path)
        if not self.class_names:
            self.class_names = server_state.supported_plants

        prediction = self.model.predict(input_data)
        label = self.class_names[np.argmax(prediction)]
        if self.enable_fallback:
            fallback_common_name, fallback_plant_name = PlantRestHandler.identify_plant(image_path)
            print(f"Fallback response is {fallback_common_name} and {fallback_plant_name}")
            label = self.compare_strings(fallback_plant_name, server_state.supported_plants)

        print(f"LABEL IS {label}")
        print(f"json output will be: \n\n {server_state.plants_info[label]}")
        return server_state.plants_info[label]