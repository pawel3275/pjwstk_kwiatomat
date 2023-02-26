import click
import os

from exceptions.exception_handler import ErroneousImagePath
from AI.image_processing import ImagePreprocessing
from AI.model_preparing import MlModel
from server.plant_rest_handler import PlantRestHandler
from server.flask_server import app

@click.command()
@click.option("--image_path", type=str, default=None, required=False, help="Path to the image.")
@click.option("--train_model", type=bool, is_flag=True, default=False, required=False, help="Bool to determine if model training should be performed.")
def run_flower_analysys(image_path, train_model):
    if train_model:
        click.echo(f"Model will be trained")
        run_model_training()
    PlantRestHandler.identify_plant(image_path)
    run_flower_recognition(image_path)

def run_model_training():
    model_handler = MlModel()
    model_handler.prepare_flower_model()


def run_flower_recognition(image_path):
    click.echo(image_path)
    try:
        if os.path.exists(image_path):
            preprocessor = ImagePreprocessing(image_path)
            preprocessor.show_image()
            pass
        else:
            raise ErroneousImagePath(image_path)
    except Exception as exc:
        click.echo(f"Script failed due to exception: {exc}")


if __name__ == "__main__":
    app.run(host="0.0.0.0") 