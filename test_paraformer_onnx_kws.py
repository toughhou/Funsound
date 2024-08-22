from funsound.utils import *
from funsound.funasr.onnx.offline.SeacoParaformer import init_model

if __name__ == "__main__":

    # 加载模型
    cfg = load_config('conf/funasr_onnx.yaml')
    am_model = init_model(asr_model_name = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                          cfg=cfg)

    with open('keywords.txt', 'r', encoding='utf-8') as f:
        WORDS = f.readlines()
        WORDS = [line.strip() for line in WORDS if len(line.strip()) <= 10]


    audio_file = "dataset/keywords/chen_继续播放视频_00.wav"
    audio_data = read_audio_file(audio_file)
    audio_list = [audio_data]
    # results = am_model.kws(audio_list,WORDS,as_hotwords=True)
    results = am_model.kws_dtw(audio_list,WORDS,as_hotwords=True)
    pprint(results[0])

    