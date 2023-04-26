

import sys

sys.path.append("..")

import json
import logging
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import BadRequest, NotFound
from werkzeug.utils import secure_filename

from AI.ai_handler import AiHandler
from server.server_state import ServerState

from .image_downloader import ImageDownloader

UPLOAD_FOLDER = Path(Path.cwd() / "file_uploads")
ALLOWED_EXTENSIONS = {"png", "jpeg", "jpg"}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
server_state = ServerState()


def allowed_file(filename):
	"""
    Check if the filename is allowed for upload based on its extension.

    Args:
        filename (str): Name of the file to be checked.

    Returns:
        True if the file has an allowed extension, False otherwise.
    """
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles file uploads from clients and returns predictions.

    The function expects a file to be uploaded via a POST request. If the file is 
    not found or is not of an allowed file type, a JSON error response with an 
    appropriate status code is returned. If the file is valid, it is saved to the 
    server's UPLOAD_FOLDER directory and then passed to an AiHandler object to 
    generate a prediction. The prediction result is returned to the client as a JSON 
    response with a success status code.

    Returns:
        A JSON response containing a message and a status code indicating the 
        success or failure of the request.
    """
    if 'file' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp

    file = request.files['file']

    if file.filename == '':
        resp = jsonify({'message' : 'No file selected for uploading'})
        resp.status_code = 400
        return resp

    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            path_for_image = Path(app.config['UPLOAD_FOLDER'] / filename)
            file.save(path_for_image)

            handler = AiHandler()
            response = handler.predict_image(server_state, path_for_image)

            path_for_image.unlink()
            resp = jsonify({'message' : response})
            resp.status_code = 201
            return resp

        except Exception as e:
            resp = jsonify({'message' : f'An error occurred while processing the image: {str(e)}'})
            resp.status_code = 500
            return resp
    else:
        resp = jsonify({'message' : f'Allowed file types are txt, pdf, png, jpg, jpeg, gif'})
        resp.status_code = 400
        return resp


@app.route('/download-training-data', methods=["GET"])
def download_training_data():
    """
    Downloads images for training data of supported plants.

    Returns:
        A JSON response containing a message and a status code indicating the
        success or failure of the request.
    """
    try:
        downloader = ImageDownloader()
        downloader.download_all_images()
        server_state.set_supported_plants(downloader.supported_plants)
        server_state.set_plants_info(downloader.plants_info)
        resp = jsonify({'message': 'Images successfully downloaded'})
        resp.status_code = 201
        return resp
    except BadRequest as e:
        logging.exception(e)
        return jsonify({'error': str(e)}), 400

    except NotFound as e:
        logging.exception(e)
        return jsonify({'error': str(e)}), 404

    except Exception as e:
        logging.exception(e)
        return jsonify({'error': 'An unexpected error occurred.'}), 500


@app.route('/train-model', methods=["GET"])
def preprocess_data():
    """
    Triggers the preprocessing and training of the machine learning model.

    Returns:
        str: A JSON response indicating that the data has been preprocessed.
    """
    try:
        handler = AiHandler()
        handler.preprocess_images()
        handler.train_model()
        server_state.set_ai_model_classes(handler.class_names)
        resp = jsonify({'message': 'All data preprocessed.'})
        resp.status_code = 201
        return resp

    except BadRequest as e:
        logging.exception(e)
        return jsonify({'error': str(e)}), 400

    except NotFound as e:
        logging.exception(e)
        return jsonify({'error': str(e)}), 404

    except Exception as e:
        logging.exception(e)
        return jsonify({'error': 'An unexpected error occurred.'}), 500


@app.route('/predict-beit', methods=["POST"])
def predict_beit():
    """
    Route function to predict image using the trained AI model.

    Returns:
        str: A JSON object with a success message and a status code.
    """
    if 'file' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp

    file = request.files['file']

    if file.filename == '':
        resp = jsonify({'message' : 'No file selected for uploading'})
        resp.status_code = 400
        return resp

    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            path_for_image = Path(app.config['UPLOAD_FOLDER'] / filename)
            file.save(path_for_image)

            handler = AiHandler()
            response = handler.predict_image(server_state, path_for_image, use_beit=True)

            path_for_image.unlink()
            resp = jsonify({'message' : response})
            resp.status_code = 201
            return resp

        except Exception as e:
            resp = jsonify({'message' : f'An error occurred while processing the image: {str(e)}'})
            resp.status_code = 500
            return resp
    else:
        resp = jsonify({'message' : f'Allowed file types are {ALLOWED_EXTENSIONS}'})
        resp.status_code = 400
        return resp


@app.route('/health-check', methods=["GET"])
def health_check():
    """
    Function that returns simple status to check if the server is up and running

    Returns:
        str: A JSON object with a success message and a status code.
    """
    try:
        # Check any dependencies or connections needed to make the server functional
        # Here, you can add any custom checks that need to be performed to ensure the server is healthy

        # Return a success message if the server is running
        data = server_state.data_downloaded
        model = server_state.model_trained
        directories = server_state.directories_created
        output = f"""
            Server is up and running.

            Current status:
            data downloaded: {data}
            model tranined: {model}
            directories created: {directories}
        """
        resp = jsonify({'message': output})
        resp.status_code = 200
        return resp

    except BadRequest as e:
        logging.exception(e)
        return jsonify({'error': str(e)}), 400

    except NotFound as e:
        logging.exception(e)
        return jsonify({'error': str(e)}), 404

    except Exception as e:
        logging.exception(e)
        return jsonify({'error': 'An unexpected error occurred.'}), 500

@app.route('/get-flowers-info', methods=["GET"])
def get_flowers_info():
    """
    Function that returns simple status to check if the server is up and running

    Returns:
        str: A JSON object with a success message and a status code.
    """
    try:
        # Validate any parameters that are passed to the endpoint
        if request.args.get('format') not in ['json', 'xml']:
            raise BadRequest('Invalid format specified.')

        # Check if the server_state.plants_info object exists
        if not server_state.plants_info:
            raise NotFound('The requested resource was not found.')

        # Convert server_state.plants_info object to a string
        stringed_info = json.dumps(server_state.plants_info)

        # Return the data as a JSON object
        return jsonify({'plants_info': stringed_info}), 200

    except BadRequest as e:
        logging.exception(e)
        return jsonify({'error': str(e)}), 400

    except NotFound as e:
        logging.exception(e)
        return jsonify({'error': str(e)}), 404

    except Exception as e:
        logging.exception(e)
        return jsonify({'error': 'An unexpected error occurred.'}), 500


@app.route('/')
def main_page():
    """
    Show main page for kwiatomat with server_state as dict generated via the index.html in templates.

    Returns:
        str: html code with webpage
    """
    try:
        state = server_state.get_server_state()
        return render_template('index.html', result=state)

    except Exception as e:
        logging.exception(e)
        return jsonify({'error': 'An unexpected error occurred.'}), 500


