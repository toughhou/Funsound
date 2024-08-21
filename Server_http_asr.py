from flask import Flask, request, jsonify, render_template, send_from_directory
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



