import glob
import json
from pathlib import Path


class ServerState():
    """
    This class represents the current state of the server. It includes information such as whether the data is downloaded,
    model is trained, directories are created, and other relevant information.
    """

    data_downloaded = False
    model_trained = False
    directories_created = False
    model_path = None
    supported_plants = None
    ai_model_classes = None
    plants_info = None
    model_dir = Path(Path.cwd() / "models/")
    dataset_dir = Path(Path.cwd() / "dataset/")
    upload_dir = Path(Path.cwd() / "file_uploads/")

    def __init__(self) -> None:
        """
        Initializes the instance of the ServerState class. Checks if the folder structure is already created and creates the folder structure if it is not created. 
        Checks if the model is trained and if the data is downloaded. Sets the ai_model_classes and plants_info attributes.
        """
        self.check_folder_structure()
        if not self.directories_created:
            self.create_folder_structure()
            self.directories_created = True
        if not self.model_trained:
            self.check_if_model_trained()
        if not self.data_downloaded:
            self.check_if_dataset_downloaded()
        self.set_ai_model_classes(None)
        self.set_plants_info(None)

    @classmethod
    def get_server_state(cls):
        """
        Returns a dictionary containing the current state of the server, including whether the data is downloaded, 
        model is trained, directories are created, and other relevant information.
        """
        cls.check_if_dataset_downloaded()
        cls.check_if_model_trained()
        cls.load_supported_plants()
        return {
            "data_downloaded:": cls.data_downloaded,
            "model_trained:": cls.model_trained,
            "directories_created:": cls.directories_created,
            "model_path:": cls.model_path,
            "model_dir:": cls.model_dir,
            "dataset_dir:": cls.dataset_dir,
            "upload_dir:": cls.upload_dir,
            "supported_plants": cls.supported_plants,
            "plants_info": cls.plants_info
        }

    @classmethod
    def check_if_model_trained(cls):
        """
        Checks whether the model has already been trained and saved, and sets the class
        variable `model_path` accordingly.

        If `model_path` is already set, the method will simply print a message indicating
        that the model is already in use.

        If `model_path` is not set, the method will search for saved models in the `model_dir`
        directory and set `model_path` to the path of the first model found. If no models are
        found, the method will print a message indicating that the directory is empty.

        Args:
            None

        Returns:
            None
        """
        if not cls.model_path:
            path = cls.model_dir
            models = list(path.glob("*.h5"))  # Filter only files with .h5 extension
            if models:
                cls.model_path = models[0]
                cls.model_trained = True
            else:
                print(f"No .h5 model files found in directory: {path}")
        else:
            print(f"Model already in use: {cls.model_path}")

    @classmethod
    def check_if_dataset_downloaded(cls):
        """
        Checks if the dataset was already downloaded to a destination folder containing training images.

        Args:
            None

        Returns:
            None
        """
        number_of_files = len(list(Path(cls.dataset_dir).rglob("*")))
        if number_of_files > 20:
            cls.data_downloaded = True

    @classmethod
    def set_model_trained(cls, value):
        """
        Sets the class attribute 'model_trained' to the given value.

        Args:
            value (bool): The value to set for the class attribute 'model_trained'.

        Returns:
            None
        """
        cls.model_trained = value

    @classmethod
    def set_data_downloaded(cls, value):
        """
        Sets the class attribute 'data_downloaded' to the given value.

        Args:
            value (bool): The value to set for the class attribute 'data_downloaded'.

        Returns:
            None
        """
        cls.data_downloaded = value

    @classmethod
    def set_supported_plants(cls, value):
        """
        Sets the class attribute 'supported_plants' to the given value.

        Args:
            value (list): The value to set for the class attribute 'supported_plants'.

        Returns:
            None
        """
        cls.supported_plants = value

    @classmethod
    def set_ai_model_classes(cls, value):
        """
        Sets the class attribute 'ai_model_classes' to the given value.
        If the value is an empty list, it tries to load the classes from a file.

        Args:
            value (list): The value to set for the class attribute 'ai_model_classes'.

        Returns:
            None
        """
        if not value:
            file = Path(Path.cwd() / "model_labels.txt")
            if file.exists():
                with open(file, "r") as input:
                    contents = input.read()
                cls.ai_model_classes = contents.replace("\'", "").split(',')
                print(
                    f"Server state loaded ai model classes as {cls.ai_model_classes}")
                cls.set_supported_plants(cls.ai_model_classes)
        cls.ai_model_classes = value

    @classmethod
    def set_plants_info(cls, value):
        """
        Sets the class attribute 'plants_info' to the given value.
        If the value is None, it tries to load the plants info from a file.

        Args:
            value (dict): The value to set for the class attribute 'plants_info'.

        Returns:
            None
        """
        if not value:
            file = Path(Path.cwd() / "supported_plants_list.json")
            if file.exists():
                with open(file, "r") as input:
                    cls.plants_info = json.loads(input.read())
                print("Plants info was loaded from file.")
        else:
            cls.plants_info = value

    @classmethod
    def check_folder_structure(cls):
        """
        Checks if the required folder structure exists and sets a flag
        to True if all directories are present.

        Args:
            None

        Returns:
            None
        """
        cls.directories_created = (
            cls.check_if_folder_exists(cls.model_dir) and
            cls.check_if_folder_exists(cls.dataset_dir) and
            cls.check_if_folder_exists(cls.upload_dir)
        )

    @classmethod
    def create_folder_structure(cls):
        """
        Creates the required folder structure if it does not exist yet.

        Args:
            None

        Returns:
            None
        """
        # Make directory containing models
        cls.create_folder(cls.model_dir)
        # Make directory containing dataset for model
        cls.create_folder(cls.dataset_dir)
        # Make directory containing user input images
        cls.create_folder(cls.upload_dir)

    @classmethod
    def load_supported_plants(cls):
        with open('supported_plants_list.json', 'r') as file:
            cls.plants_info = json.load(file)
        cls.supported_plants = list(cls.plants_info.keys())

    @staticmethod
    def check_if_folder_exists(folder_path):
        """
        Checks if a folder exists at the specified path.

        Args:
            folder_path (pathlib.Path): The path to check.

        Returns:
            bool: True if a folder exists at the specified path, False otherwise.
        """
        return folder_path.is_dir()

    @staticmethod
    def create_folder(folder_path):
        """
        Creates a folder at the specified path if it does not already exist.

        Args:
            folder_path (pathlib.Path): The path at which to create the folder.

        Returns:
            None
        """
        if not (Path.cwd() / folder_path).is_dir():
            path = Path(Path.cwd() / folder_path)
            Path.mkdir(path)
            print(f"Folder {path} successfully created!")
        else:
            print(f"Folder {folder_path} already exists!")
