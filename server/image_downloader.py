import imghdr
import json
import os
import shutil
import time
from pathlib import Path

# from PIL import Image
import PIL
from bing_image_downloader import downloader
from duckduckgo_search import \
    ddg_images  # https://pypi.org/project/duckduckgo-search/


class ImageDownloader:
    def __init__(self) -> None:
        self.dataset_folder = "dataset"
        self.plant_query_suffix = "plant"
        self.image_limit = 100
        self.timeout = 2
        self.force_use_bing = False
        self.search_queries = self._load_plants_list()
        self.supported_formats = [".JPG", ".JPEG", ".PNG"]
        self.plants_info = None
        self.supported_plants = None

    def _load_plants_list(self):
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
        response = ddg_images(
            query, 
            safesearch='Off', 
            size="medium",
            type_image="photo", 
            max_results=self.image_limit, 
            download=True
        )
        if not response:
            pass  # TODO: throw exception here
        folder_str = f"ddg_images_{query}"
        folder_path = [f for f in Path.cwd().iterdir() if folder_str in f.stem]
        if folder_path:
            for each_file in Path(folder_path[0]).glob('*.*'): # grabs all files
                trg_path = each_file.parent.parent # gets the parent of the folder 
                trg_path = Path(trg_path / self.dataset_folder / query)
                trg_path.mkdir(parents=True, exist_ok=True)
                try:
                    each_file.rename(trg_path.joinpath(each_file.name)) # moves to parent folder.
                except Exception as exc:
                    each_file.unlink()
            shutil.rmtree(folder_path[0])
        else:
            print(f"Folder {folder_str} not found")
            pass  # TODO: throw exception here
    
    def remove_unwanted_files(self):
        for path in Path('dataset').rglob('*'):
            if str(path.suffix).upper() not in self.supported_formats and not Path(path).is_dir():
                path.unlink()

    def download_all_images(self,):
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
        image_extensions = self.supported_formats  # add there all your images file extensions
        img_type_accepted_by_tf = [".bmp", ".gif", ".jpeg", ".png"]
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
