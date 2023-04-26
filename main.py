import os

from server.flask_server import app

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
if __name__ == "__main__":
    app.run(host="0.0.0.0") 
