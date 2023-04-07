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
    """
    The `AiHandler` class is responsible for managing the AI model training and prediction.
    """

    def __init__(self) -> None:
        self.enable_fallback = False
        self.model_preprocessor = MlModel()
        self.model_path = None
        self.validation_data = None
        self.training_data = None
        self.class_names = None
        self.model = None

    def preprocess_images(self):
        """
        Preprocesses the images for use in training the machine learning model.

        Returns:
            None.
        """
        self.image_preprocessor = ImagePreprocessing()
        self.training_data, self.validation_data = self.image_preprocessor.preprocess_data()
        self.class_names = self.image_preprocessor.class_names

    def train_model(self):
        """
        Trains the machine learning model using the preprocessed data.

        Returns:
            None.
        """
        model_instance = MlModel()
        self.model = model_instance.prepare_model(
            self.training_data,
            self.validation_data,
            self.class_names
        )
        timestr = time.strftime("%Y%m%d-%H%M%S")
        self.save_model(Path(Path.cwd() / "models" / f"model_{timestr}.h5"))

    def save_model(self, path):
        """
        Saves the trained machine learning model to the specified path.

        Args:
            path (str): The path to save the trained machine learning model to.

        Returns:
            None.
        """
        self.model.save(path)

    def load_model(self, path):
        """
        Loads a trained machine learning model from the specified path.

        Args:
            path (str): The path to load the trained machine learning model from.

        Returns:
            None.
        """
        self.model = tf.keras.models.load_model(path)

    def compare_strings(self, target_string, string_list):
        """
        Compare a target string with a list of strings and return the best match.

        Args:
            target_string (str): The string to be compared.
            string_list (list): A list of strings to compare the target string with.

        Returns:
            str: The best match found in the list of strings. If the best match has a similarity score of less than 0.6, the target string is returned instead.

        """
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

    def predict_image(self, server_state, image_path, use_beit=False):
        """
        Predicts the type of plant from an input image.

        Args:
            server_state (ServerState): The current state of the server.
            image_path (str): The file path of the input image.

        Returns:
            dict: A dictionary containing information about the predicted plant. If the plant is in the server state's
                database, the dictionary will contain information such as the plant's name, scientific name, and description.
                Otherwise, the dictionary will only contain the name of the predicted plant.

        """
        image_preprocessor = ImagePreprocessing()
        input_data = image_preprocessor.load_image(image_path)
        if not self.model:
            self.load_model(server_state.model_path)
        if not self.class_names:
            self.class_names = server_state.supported_plants

        if use_beit:
            label = self.model_preprocessor.use_beit_model(image_path=image_path)
        else:
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

