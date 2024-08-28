#SDK模型下载
from modelscope import snapshot_download
model_dir = snapshot_download("pengzhendong/faster-whisper-large-v2",cache_dir='./funasr_models')

