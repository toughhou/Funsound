from funsound.utils import *
from funsound.funasr.onnx.offline.SeacoParaformer import init_model

if __name__ == "__main__":

    # 加载模型
    cfg = load_config('funsound/conf/funasr_onnx.yaml')
    am_model = init_model(cfg=cfg)

    # data_dir = "dataset/keywords"
    data_dir = "dataset/keywords_snr=5"

    utt_list, audio_list, trans_list = load_dataset(data_dir,has_trans=False)

    with open('keywords.txt', 'r', encoding='utf-8') as f:
        WORDS = f.readlines()
        WORDS = [line.strip() for line in WORDS if len(line.strip()) <= 10]

    results = am_model.kws(audio_list,WORDS,as_hotwords=True)
    # results = am_model.kws_dtw(audio_list,WORDS,as_hotwords=True)

    success = 0
    for utt, result in zip(utt_list,results):
        ref = utt.split("_")[1]
        hyp = result[0][0]
        score = result[0][1]
        success += ref==hyp

        audio_file = "%s/%s.wav"%(data_dir,utt)
        if ref!=hyp:
            print(utt, result, audio_file)
    print(len(utt_list),success, success/len(utt_list))