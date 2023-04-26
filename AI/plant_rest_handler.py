import base64

import requests


class PlantRestHandler():
    """
    A class for handling REST API calls for plant identification using the Plant.id API.
    """
    def __init__(self) -> None:
        """
        Constructs a new instance of the `PlantRestHandler` class.
        """
        pass
    
    @staticmethod
    def identify_plant(image_path):
        """
        Identifies a plant from the image located at `image_path` using the Plant.id API.
        
        Args:
        image_path (str): The path of the image to be identified.
        
        Returns
            Tuple: A tuple containing the common name and scientific name of the identified plant.
        
        Raises
            JSONDecodeError
                If the response from the Plant.id API is not in JSON format.
            HTTPError
                If the request to the Plant.id API returns an unsuccessful status code.
        
        """
        with open(image_path, "rb") as file:
            images = [base64.b64encode(file.read()).decode("ascii")]
        response = requests.post(
            "https://api.plant.id/v2/identify",
            json={
                "images": images,
                "modifiers": ["similar_images"],
                "plant_details": ["common_names", "url"],
            },
            headers={
                "Content-Type": "application/json",
                "Api-Key": "Hh6ioRt9mvdw07XdjifsUlXspgFyX9jLREfCRWkZMbcUscGqJn",
        }).json()
        
        best_match = response["suggestions"][0]
        common_name = best_match["plant_details"]["common_names"]
        plant_name = best_match["plant_name"]
        return common_name, plant_name