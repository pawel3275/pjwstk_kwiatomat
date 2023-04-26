class ErroneousImagePath(Exception):
    """
    Exception raised when an image path provided is invalid or does not exist.

    Attributes:
    - image_path: The path of the image that caused the exception.
    """
    def __init__(self, image_path) -> None:
        self.message = f"Image path {image_path} is invalid or image does not exist."
        super().__init__(self.message)

class DdgDownloaderError(Exception):
    """
    Exception raised when there is an error while downloading images using DDG.

    Attributes:
    - response: The response received from DDG that caused the exception.
    """
    def __init__(self, response) -> None:
        self.message = f"Unable to download images using DDG. Response: {response}"
        super().__init__(self.message)

class MissingFolderException(Exception):
    """
    Exception raised when a folder is not found.

    Attributes:
    - folder_path: The path of the folder that was not found.
    """
    def __init__(self, folder_path) -> None:
        self.message = f"Folder {folder_path} was not found."
        super().__init__(self.message)
