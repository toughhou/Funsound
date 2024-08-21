from flask import Flask, request, jsonify, render_template, send_from_directory, make_response
from werkzeug.utils import secure_filename
from funsound.whisper.asr import ASR 
from funsound.common.executor import *


def init_engine():
    engine = ASR(model_id='/opt/wangwei/funsound_onnx/funasr_models/keepitsimple/faster-whisper-large-v3',
                cfg_file='conf/whisper.yaml',
                log_file=f'log/whisper-{id}.log')
    engine.init_state()
    return engine

def processor(self,params):
    audio_file = params[0]
    result = self.engine.inference(audio_file)
    return result
Worker.processor = processor

app = Flask(__name__)

RESPONSE_TEMPLATE = {
    "code": 1, 
    "message": "",
    'asr': {
        "result": []
    }
}

def create_response(content):
    response = make_response(content)
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response

@app.route('/submit_task', methods=['POST'])
def submit_task():
    response_data = RESPONSE_TEMPLATE.copy()

    if 'file' not in request.files:
        response_data['message'] = "error: No file part in the request"
        return create_response(jsonify(response_data))
    



