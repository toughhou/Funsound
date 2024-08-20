from funsound.whisper.asr import ASR 


asr_ = ASR(model_id='/opt/wangwei/funsound_onnx/funasr_models/keepitsimple/faster-whisper-large-v3',
           cfg_file='conf/whisper.yaml',
           log_file='log/whisper.log')
asr_.init_state()

res = asr_.inference('/opt/wangwei/funsound_onnx/funsound/examples/test1.wav')
print(res)
