from flask import Flask, request, jsonify, make_response, render_template
from werkzeug.utils import secure_filename
from funsound.common.executor import *
import os

"""
初始化语音识别引擎，加载模型和配置文件，并初始化状态。
"""
# def init_engine(id):
#     from funsound.whisper.asr import ASR 
#     engine = ASR(
#                  cfg_file='conf/whisper.yaml',
#                  log_file=f'log/whisper-{id}.log')
#     engine.init_state()
#     return engine


def init_engine(id):
    from funsound.funasr.onnx.offline.asr import ASR
    engine = ASR(
                cfg_file='conf/funasr_onnx.yaml',
                log_file=f'log/funasr-{id}.log')
    engine.init_state()
    return engine


# 定义任务处理器
def processor(self, params):
    """
    执行音频文件的语音识别，并将结果保存到文件。
    """
    audio_file = params[0]
    result = self.engine.inference(audio_file)
    trans_file = f"{audio_file}.trans"
    with open(trans_file, 'wt') as f:
        for line in result:
            print(line, file=f)
    return result

# 将processor函数绑定到Worker类
Worker.processor = processor

# 初始化Flask应用
app = Flask(__name__)

# 全局变量
RESPONSE_TEMPLATE = {
    "code": 1, 
    "message": "",
    'content': []
}
DATADIR = "cache"
mkdir(DATADIR)  # 创建缓存目录
mkdir('log')

# 创建响应
def create_response(content):
    """
    创建一个带有JSON格式内容的响应。
    """
    response = make_response(content)
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response

# 检查文件类型是否允许
def allowed_file(filename):
    """
    检查文件是否为允许的类型。
    """
    allowed_extensions = {'wav', 'mp3', 'm4a', 'mp4', 'aac'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


@app.route('/')
def index():
    return render_template('index.html')

# 处理文件提交请求
@app.route('/submit', methods=['POST'])
def submit():
    response_data = RESPONSE_TEMPLATE.copy()

    # 检查请求中是否包含文件
    if 'file' not in request.files:
        response_data['message'] = "error: No file part in the request"
        return create_response(jsonify(response_data))
    
    audio_file = request.files['file']

    # 检查文件是否有文件名
    if audio_file.filename == '':
        response_data['message'] = "error: No selected file"
        return create_response(jsonify(response_data))

    # 检查文件类型是否允许
    if not allowed_file(audio_file.filename):
        response_data['message'] = "error: File type not allowed"
        return create_response(jsonify(response_data))
    
    # 生成随机文件名并保存文件
    audio_format = audio_file.filename.rsplit('.', 1)[1].lower()
    random_filename = get_current_time() + "-" + generate_random_string(10)
    task_file_path = os.path.join(DATADIR, f"{random_filename}.{audio_format}")
    audio_file.save(task_file_path)

    # 提交任务并返回任务ID
    task_id = submit_task(workers, params=[task_file_path])
    response_data['code'] = 0
    response_data['message'] = "success"
    response_data['content'] = task_id

    return create_response(jsonify(response_data))

# 查询任务进度
@app.route('/task_prgs/<task_id>', methods=['GET'])
def task_status(task_id):
    response_data = RESPONSE_TEMPLATE.copy()
    prgs = get_task_progress(task_id)
    
    if not prgs:
        response_data['message'] = f"error: No such task id: {task_id}"
        return create_response(jsonify(response_data))
    else:
        response_data['code'] = 0
        response_data['message'] = "success"
        response_data['content'] = prgs
        return create_response(jsonify(response_data))
    
# 查询工作者状态
@app.route('/worker_status', methods=['GET'])
def worker_status():
    response_data = RESPONSE_TEMPLATE.copy()
    response_data['code'] = 0
    response_data['message'] = "success"
    response_data['content'] = get_worker_status(workers)
    return create_response(jsonify(response_data))

# 主程序入口
if __name__ == "__main__":
    workers = []
    
    # 初始化和加载工作者
    for id in range(3):
        engine = init_engine(id)
        worker = Worker(wid=id, log_file=f'log/worker-{id}.log')
        worker.load_engine(engine=engine)
        workers.append(worker)
    
    # 启动工作者
    launch(workers)

    # 启动Flask应用，监听5003端口
    app.run("0.0.0.0", port=5001)
