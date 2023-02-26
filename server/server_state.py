from pathlib import Path
import json

class ServerState():
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
        cls.check_if_dataset_downloaded()
        cls.check_if_model_trained()
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
        if not cls.model_path:
            path = cls.model_dir
            models = list(path.iterdir())
            if models:
                cls.model_path = models[0]
                cls.model_trained = True
            else:
                print(f"Directory of {path} is empty! Iterdir returned: {models}")
        else:
            print(f"Model already in use: {cls.model_path}")

    @classmethod
    def check_if_dataset_downloaded(cls):
        number_of_files = len(list(Path(cls.dataset_dir).rglob("*")))
        print(f"{number_of_files} found in {cls.dataset_dir}")
        if number_of_files > 20:
            cls.data_downloaded = True

    @classmethod
    def set_model_trained(cls, value):
        cls.model_trained = value
    
    @classmethod
    def set_data_downloaded(cls, value):
        cls.data_downloaded = value

    @classmethod
    def set_supported_plants(cls, value):
        cls.supported_plants = value

    @classmethod
    def set_ai_model_classes(cls, value):
        if not value:
            file = Path(Path.cwd() / "model_labels.txt")
            if file.exists():
                with open(file, "r") as input:
                    contents = input.read()
                cls.ai_model_classes = contents.replace("\'", "").split(',')
                print(f"Server state loaded ai model classes as {cls.ai_model_classes}")
                cls.set_supported_plants(cls.ai_model_classes)
        cls.ai_model_classes = value

    @classmethod
    def set_plants_info(cls, value):
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
        cls.directories_created = (
            cls.check_if_folder_exists(cls.model_dir) and \
            cls.check_if_folder_exists(cls.dataset_dir) and \
            cls.check_if_folder_exists(cls.upload_dir)
        )
        print(f"Folder structure check returned {cls.directories_created}")

    @classmethod
    def create_folder_structure(cls):
        # Make directory containing models
        cls.create_folder(cls.model_dir)
        # Make directory containing dataset for model
        cls.create_folder(cls.dataset_dir)
        # Make directory containing user input images
        cls.create_folder(cls.upload_dir)
    
    @staticmethod
    def check_if_folder_exists(folder_path):
        return folder_path.is_dir()

    @staticmethod
    def create_folder(folder_path):
        if not (Path.cwd() / folder_path).is_dir():
            path = Path(Path.cwd() / folder_path)
            Path.mkdir(path)
            print(f"Folder {path} successfully created!")
        else:
            print(f"Folder {folder_path} already exists!") 
