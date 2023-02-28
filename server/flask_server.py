

import sys
sys.path.append("..")

from flask import Flask, render_template
from flask_restful import Api
from server_endpoint import plant_api

from flask import Flask, request, redirect, jsonify
from werkzeug.utils import secure_filename
import os
from exceptions.exception_handler import ErroneousImagePath
from AI.image_processing import ImagePreprocessing
from AI.model_preparing import MlModel
import logging
from .image_downloader import ImageDownloader
from server.server_state import ServerState
from AI.ai_handler import AiHandler

UPLOAD_FOLDER = 'D:/scratch/inzynierka/file_uploads'
ALLOWED_EXTENSIONS = {"png", "jpeg", "jpg"}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
server_state = ServerState()
#logging.basicConfig(filename='record.log', level=logging.DEBUG, format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/file-upload', methods=['POST'])
def upload_file():
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
		path_for_image = os.path.join(app.config['UPLOAD_FOLDER'], filename)
		file.save(path_for_image)
		print(f"File saved as {path_for_image}")

		handler = AiHandler()
		response = handler.predict_image(server_state, path_for_image)

		print(f"Response for plant recognition was {response}")
		os.remove(path_for_image)
		print(f"Everything done, removing file: {path_for_image}")
		resp = jsonify({'message' : response})
		resp.status_code = 201
		return resp
	else:
		resp = jsonify({'message' : f'Allowed file types are txt, pdf, png, jpg, jpeg, gif'})
		resp.status_code = 400
		return resp

@app.route('/download-training-data', methods=["GET"])
def download_training_data():
	downloader = ImageDownloader()
	downloader.download_all_images()
	server_state.set_supported_plants(downloader.supported_plants)
	server_state.set_plants_info(downloader.plants_info)
	resp = jsonify({'message' : 'File successfully uploaded'})
	resp.status_code = 201
	return resp

@app.route('/preprocess_data', methods=["GET"])
def preprocess_data():
	handler = AiHandler()
	handler.preprocess_images()
	handler.train_model()
	server_state.set_ai_model_classes(handler.class_names)
	resp = jsonify({'message' : 'All data preprocessed.'})
	resp.status_code = 201
	return resp

@app.route('/predict-image', methods=["GET"])
def predict_image():
	handler = AiHandler()
	handler.predict_image(server_state, "dummy_path")
	resp = jsonify({'message' : f'PREDICTION DONE'})
	resp.status_code = 200
	return resp


@app.route('/')
def main_page():
    state = server_state.get_server_state()
    return render_template('index.html',result=state)


