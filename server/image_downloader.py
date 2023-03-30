import imghdr
import json
import os
import shutil
from pathlib import Path
import PIL

from bing_image_downloader import downloader
from duckduckgo_search import \
    ddg_images  # https://pypi.org/project/duckduckgo-search/


class ImageDownloader:
    """
    A class for downloading images from the internet and preparing them for use with TensorFlow.

    Attributes:
        TF_SUPPORTED_FORMATS (list): A list of file extensions that TensorFlow supports.
    """
    TF_SUPPORTED_FORMATS = [".bmp", ".gif", ".jpeg", ".jpg", ".png"]

    def __init__(self) -> None:
        """
        Initializes a PlantImageDownloader object with default settings and loads the list of supported plants.

        Attributes:
            dataset_folder (str): The name of the folder where the downloaded images will be saved.
            plant_query_suffix (str): The suffix to be added to each plant name to form the search query.
            image_limit (int): The maximum number of images to be downloaded per plant.
            timeout (int): The number of seconds to wait before timing out the download request.
            force_use_bing (bool): Whether to force the use of Bing Images API instead of DuckDuckGo.
            search_queries (list): The list of supported plants extracted from the 'supported_plants_list.json' file.
            supported_formats (list): The list of supported image file formats.
            plants_info (dict): A dictionary containing information about each plant (e.g., common name, scientific name).
            supported_plants (list): A list of supported plants extracted from the 'supported_plants_list.json' file.
        """
        self.dataset_folder = "dataset"
        self.plant_query_suffix = "plant"
        self.image_limit = 220
        self.timeout = 2
        self.force_use_bing = False
        self.search_queries = self._load_plants_list()
        self.supported_formats = [".JPG", ".JPEG", ".PNG"]
        self.plants_info = None
        self.supported_plants = None

    def _load_plants_list(self):
        """
        Load the list of supported plants from a JSON file and extract the keys as the list of supported plants.

        Returns:
            list: A list of supported plants extracted from the JSON file.
        """
        with open('supported_plants_list.json', 'r') as file:
            self.plants_info = json.load(file)
        self.supported_plants = self.plants_info.keys()
        return self.supported_plants

    def _download_using_bing(self, query):
        status = downloader.download(
            query,
            limit=self.image_limit,
            output_dir=self.dataset_folder,
            adult_filter_off=True,
            force_replace=False,
            filter="photo",
            timeout=self.timeout)
        return status

    def _download_using_ddg(self, query):
        """Downloads images for a given query using the DuckDuckGo search engine.

        The downloaded images are saved to a folder named `query` within the `dataset` directory. Any existing
        folder with the same name is overwritten.

        Args:
            query (str): The search query to use.

        Returns:
            None
        """
        folder_name = f"ddg_images_{query}"
        folder_path = Path.cwd() / folder_name

        # Use ddg_images to download images
        response = ddg_images(
            query,
            safesearch='Off',
            size="medium",
            type_image="photo",
            max_results=self.image_limit,
            download=True,
        )

        if not response:
            pass  # TODO: throw exception here
        folder_str = f"ddg_images_{query}"
        folder_path = [f for f in Path.cwd().iterdir() if folder_str in f.stem]
        if folder_path:
            # grabs all files
            for each_file in Path(folder_path[0]).glob('*.*'):
                trg_path = each_file.parent.parent  # gets the parent of the folder
                trg_path = Path(trg_path / self.dataset_folder / query)
                trg_path.mkdir(parents=True, exist_ok=True)
                try:
                    # moves to parent folder.
                    each_file.rename(trg_path.joinpath(each_file.name))
                except Exception as exc:
                    each_file.unlink()
            shutil.rmtree(folder_path[0])
        else:
            print(f"Folder {folder_str} not found")
            pass  # TODO: throw exception here

    def remove_unwanted_files(self):
        """Removes files with unsupported file extensions from the `dataset` directory.

        This method scans the `dataset` directory and its subdirectories for files with unsupported file extensions.
        If a file has an unsupported file extension, it is deleted. If a subdirectory is encountered, it is not
        processed.

        Supported file extensions are defined by the `supported_formats` list.

        Returns:
            None
        """
        for path in Path('dataset').rglob('*'):
            if str(path.suffix).upper() not in self.supported_formats and not Path(path).is_dir():
                path.unlink()

    def download_all_images(self,):
        """Downloads all images for the specified search queries.

        If `force_use_bing` is set to `True`, this method will use the Bing image search API to download images
        for each search query. Otherwise, it will use the DuckDuckGo search engine.

        The downloaded images are saved to the `dataset` directory, which is created if it does not already exist.

        After the images are downloaded, this method calls `remove_unwanted_files` to remove any files with
        unsupported file extensions from the `dataset` directory. It then calls `check_tensorflow_compatibility`
        to prepare the dataset for TensorFlow compatibility.

        Returns:
            A string indicating the status of the download process. If all images were successfully downloaded,
            "succ" is returned. Otherwise, "fail" is returned.
        """
        status = "fail"
        if self.force_use_bing:
            for query in self.search_queries:
                status = self._download_using_bing(query)
        else:
            for query in self.search_queries:
                self._download_using_ddg(query)
                status = "succ"
        self.remove_unwanted_files()
        self.check_tensorflow_compatibility()
        return status

    def check_tensorflow_compatibility(self):
        """Prepares the image dataset for TensorFlow compatibility.

        This method performs the following tasks on the image dataset:
        - Removes unsupported file formats
        - Removes images smaller than 10KB
        - Verifies that all images can be opened and read
        - Converts images to JPEG format if necessary

        The modified dataset is stored in the same directory as the original dataset.
        """
        image_extensions = self.supported_formats  # add there all your images file extensions
        for filepath in Path(self.dataset_folder).rglob("*"):
            if filepath.suffix.upper() in image_extensions:
                if os.path.getsize(filepath) < 10 * 1024:
                    filepath.unlink()
                    continue
                try:
                    img = PIL.Image.open(filepath)
                    img.verify() #I perform also verify, don't know if he sees other types o defects
                    img.close() #reload is necessary in my case
                    img = PIL.Image.open(filepath) 
                    img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
                    img.close()
                except Exception as exc:
                    if filepath.exists():
                        filepath.unlink()
                    continue
                img_type = imghdr.what(filepath)
                if img_type is None:
                    filepath.unlink()
                    continue
        conversion_coutner = 0
        for filepath in Path(self.dataset_folder).rglob("*"):
            if filepath.suffix.upper() == ".JPG" or filepath.suffix.upper() == ".PNG":
                img = PIL.Image.open(filepath)
                rgb_im = img.convert('RGB')
                rgb_im.save(f"{filepath.parent}/converted_file_{conversion_coutner}.jpeg")
                rgb_im.close()
                img.close()
                conversion_coutner+=1
                filepath.unlink()
