

import sys

sys.path.append("..")

from pathlib import Path

from flask import Flask, jsonify, render_template, request
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

@app.route('/file-upload', methods=['POST'])
def upload_file():
	"""Handles file uploads from clients and returns predictions.

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
		filename = secure_filename(file.filename)
		path_for_image = Path(app.config['UPLOAD_FOLDER'] / filename)
		file.save(path_for_image)
		handler = AiHandler()
		response = handler.predict_image(server_state, path_for_image)
		path_for_image.unlink()
		resp = jsonify({'message' : response})
		print(f"Detected plant:\n {resp}")
		resp.status_code = 201
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
	downloader = ImageDownloader()
	downloader.download_all_images()
	server_state.set_supported_plants(downloader.supported_plants)
	server_state.set_plants_info(downloader.plants_info)
	resp = jsonify({'message' : 'File successfully uploaded'})
	resp.status_code = 201
	return resp

@app.route('/train-model', methods=["GET"])
def preprocess_data():
	"""
    Triggers the preprocessing and training of the machine learning model.

    Returns:
        str: A JSON response indicating that the data has been preprocessed.
    """
	handler = AiHandler()
	handler.preprocess_images()
	handler.train_model()
	server_state.set_ai_model_classes(handler.class_names)
	resp = jsonify({'message' : 'All data preprocessed.'})
	resp.status_code = 201
	return resp

@app.route('/predict-image', methods=["GET"])
def predict_image():
	"""
	Route function to predict image using the trained AI model.

	Returns:
		str: A JSON object with a success message and a status code.
	"""
	handler = AiHandler()
	handler.predict_image(server_state, "dummy_path")
	resp = jsonify({'message' : f'PREDICTION DONE'})
	resp.status_code = 200
	return resp


@app.route('/')
def main_page():
    """
	Show main page for kwiatomat with server_state as dict generated via the index.html in templates.

	Returns:
		str: html code with webpage
	"""
    state = server_state.get_server_state()
    return render_template('index.html',result=state)


