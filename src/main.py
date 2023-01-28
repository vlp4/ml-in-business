import os
import sys

import dill
import flask
import pandas as pd

from model import make_model

import logging

# Initialize logging
root = logging.getLogger()
root.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)
logger = logging.getLogger(__name__)

# Initialize APi
app = flask.Flask(__name__)
g_model = None


def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = dill.load(f)
    return model


def save_model(model, model_path):
    with open(model_path, 'wb') as f:
        dill.dump(model, f)



@app.route("/", methods=["GET"])
def general():
    return """Welcome to the prediction process. Please use 'http://<address>/predict' to POST"""


@app.route("/predict", methods=["POST"])
def predict():
    logger.info(f'/predict invoked')
    global g_model
    response = {
        'success': False,
        'prediction': None,
    }

    request = flask.request.get_json()
    logger.info(f'predict() request is {request}')
    try:
        data = {k: [v] for k, v in request.items()}
        df = pd.DataFrame(data)
        pred = g_model.predict_proba(df)
        response['prediction'] = pred[:, 1][0]
        response['success'] = True
    except AttributeError as e:
        logger.warning(f'Exception: {str(e)}')
        response['error'] = str(e)

    logger.info(f'predict() response is {response}')
    return flask.jsonify(response)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":

    model_file = os.environ.get('MODEL', 'models/model.dill')

    command = sys.argv[1] if len(sys.argv) > 1 else None
    if command == 'model':
        logger.info(f'Creating model...')
        model = make_model()
        logger.info(f'Saving model to {model_file}')
        save_model(model, model_file)

    else:
        host = os.environ.get('HOST', '0.0.0.0')
        port = int(os.environ.get('PORT', 8180))

        logger.info(f'Loading model from {model_file}')
        g_model = load_model(model_file)

        logger.info(f'Running model server at http://{host}:{port}')
        app.run(host=host, debug=True, port=port)
