from funsound.utils import *
from funsound.onnx.offline.SeacoParaformer import init_model
from funsound.compute_mer import compute_mer_text

if __name__ == "__main__":

    # 加载模型
    am_model = init_model(asr_model_name = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                          cfg_file='conf/onnx.yaml')

    # 加载音频
    audio_file = 'funsound/examples/test1.wav'
    trans_file = "funsound/examples/test1.trans"
    audio_list = read_audio_with_split(audio_file,window_seconds=30)

    # 语音识别
    results, *p = am_model(audio_list)
    for line in results:
        print(line)

    # 计算字错率
    with open(trans_file,'rt') as f:
        audio_trans = f.read()
        mer = compute_mer_text(audio_trans, "".join(results))
        print("Mix Error Rate:", mer)