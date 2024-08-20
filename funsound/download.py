#SDK模型下载
from modelscope import snapshot_download
model_dir = snapshot_download("keepitsimple/faster-whisper-large-v3",cache_dir='./funasr_models')

