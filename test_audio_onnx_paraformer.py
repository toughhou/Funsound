from funsound.utils import *
from funsound.onnx.SeacoParaformer import SeacoParaformerPlus, init_model
from funsound.compute_mer import compute_mer_text
import sys 

if __name__ == "__main__":

    am_model = init_model(asr_model_name = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")

    audio_file = '/opt/wangwei/funsound_onnx/funsound/examples/test1.wav'
    trans_file = "/opt/wangwei/funsound_onnx/funsound/examples/test1.trans"
    audio_list = read_audio_with_split(audio_file,window_seconds=30)

    results, *p = am_model(audio_list)
    results = "".join(results)

    with open(trans_file,'rt') as f:
        audio_trans = f.read()
        mer = compute_mer_text(audio_trans, results)
        print("Mix Error Rate:", mer)