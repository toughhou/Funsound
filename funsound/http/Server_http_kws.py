from flask import Flask, request, jsonify, make_response
from funsound.utils import *
from funsound.funasr.onnx.offline.SeacoParaformer import init_model

RESPONSE_TEMPLATE = {
    "code": 1, 
    "message": "",
    'kws': {
        "cost_audio_seconds": 0,
        "cost_decoding_seconds": 0,
        "result": []
    }
}

app = Flask(__name__)

def create_response(content):
    response = make_response(content)
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response

@app.route('/kws', methods=['POST'])
def recognition():
    response_data = RESPONSE_TEMPLATE.copy()

    client_ip = request.remote_addr
    print("Client IP:", client_ip)
    
    # 从请求中读取音频文件
    file = request.files.get('file')
    if not file:
        response_data['message'] = "No file provided"
        return create_response(jsonify(response_data))
    
    audio_bytes = file.read()
    
    # 使用 ffmpeg 处理音频
    try:
        with Timer() as t:
            pcm_data = audio_i2f(read_audio_bytes(audio_bytes))
        response_data['kws']["cost_audio_seconds"] = t.interval
    except Exception as e:
        response_data['message'] = str(e)
        return create_response(jsonify(response_data))

    # 使用模型进行 ASR
    try:
        with Timer() as t:
            results = am_model.kws([pcm_data], WORDS)
        response_data['kws']["cost_decoding_seconds"] = t.interval
        response_data['kws']['result'] = (float(results[0][0]),results[0][1])
    except Exception as e:
        response_data['message'] = str(e)
        return create_response(jsonify(response_data))

    # 创建并返回 JSON 响应
    response_data['code'] = 0
    response_data['message'] = "success"
    print(response_data)
    
    return create_response(jsonify(response_data))

if __name__ == '__main__':
    cfg = load_config('conf/funasr_onnx.yaml')
    keywords_file = cfg['KWS']['keywords_file']
    host = cfg['HTTP']['host']
    port = cfg['HTTP']['port']

    am_model = init_model(asr_model_name="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch", cfg=cfg)
    with open(keywords_file, 'r', encoding='utf-8') as f:
        WORDS = [line.strip() for line in f if len(line.strip()) <= 10]

    app.run(host=host, port=port, debug=False)
