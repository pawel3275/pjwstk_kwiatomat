


class ErroneousImagePath(Exception):
    def __init__(self, image_path) -> None:
        self.image_path = image_path
        self.message = f"Image path {image_path} is invalid or image does not exist."
        super().__init__(self.message)