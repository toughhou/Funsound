from funsound.utils import *
from funsound.onnx.offline.SeacoParaformer import init_model

if __name__ == "__main__":

    # 加载模型
    cfg = load_config('conf/onnx.yaml')
    am_model = init_model(asr_model_name = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                          cfg=cfg)

    data_dir = "/opt/wangwei/funsound_onnx/dataset/keywords"

    utt_list, audio_list, trans_list = load_dataset(data_dir,has_trans=False)

    with open('keywords.txt', 'r', encoding='utf-8') as f:
        WORDS = f.readlines()
        WORDS = [line.strip() for line in WORDS if len(line.strip()) <= 10]

    results = am_model.kws(audio_list,WORDS)
    success = 0
    for utt, result in zip(utt_list,results):
        ref = utt.split("_")[1]
        hyp = result[1]
        score = result[0]
        success += ref==hyp
        print(utt,hyp,score)
    print(len(utt_list),success/len(utt_list))