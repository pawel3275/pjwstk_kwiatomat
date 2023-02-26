import base64
import requests

class PlantRestHandler():
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def identify_plant(image_path):
        with open(image_path, "rb") as file:
            images = [base64.b64encode(file.read()).decode("ascii")]
        print("marker")
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
        print(response)
        for suggestion in response["suggestions"]:
            print(suggestion["plant_name"])    # Taraxacum officinale
            print(suggestion["plant_details"]["common_names"])    # ["Dandelion"]
            print(suggestion["plant_details"]["url"])    # https://en.wikipedia.org/wiki/Taraxacum_officinale