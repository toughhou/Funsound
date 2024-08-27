from funsound.utils import *
from funsound.funasr.onnx.offline.SeacoParaformer import init_model
from funsound.compute_mer import compute_mer_text

if __name__ == "__main__":

    # 加载模型
    cfg = load_config('conf/funasr_onnx.yaml')
    am_model = init_model(asr_model_name = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                          cfg=cfg)

    # 加载音频
    audio_file = 'funsound/examples/test1.wav'
    trans_file = "funsound/examples/test1.trans"
    window_sencond = 30
    audio_list = read_audio_with_split(audio_file,window_seconds=window_sencond)

    # 语音识别
    results, timestamps, *p = am_model(audio_list)
    timestamps2 = []
    for i, timestamp in enumerate(timestamps):
        for line in timestamp:
            line[1] += i*window_sencond
            line[2] += i*window_sencond
            timestamps2.append(line)
    
    sentences = am_model.make_sentence_by_sil(timestamps2)
    for line in sentences:
        print(line)


    # 计算字错率
    with open(trans_file,'rt') as f:
        audio_trans = f.read()
        mer = compute_mer_text(audio_trans, "".join([sentence['text' ] for sentence in sentences]))
        print("Mix Error Rate:", mer)