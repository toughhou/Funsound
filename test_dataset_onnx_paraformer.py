from funsound.utils import *
from funsound.onnx.offline.SeacoParaformer import SeacoParaformerPlus, init_model
from funsound.compute_mer import compute_min_edit_distance

def test_datadir(am_model, data_dir,audio_format = 'wav'):
    utt_list, audio_list, trans_list = load_dataset(data_dir,has_trans=True)

    results, *p = am_model(audio_list)
    Num, Den = 1, 1
    for result, trans_data in zip(results,trans_list):
        med, den = compute_min_edit_distance(trans_data, result,show=False)
        Num += med
        Den += den
    return Num, Den

if __name__ == "__main__":

    # 加载模型
    cfg = load_config('conf/onnx.yaml')
    am_model = init_model(asr_model_name = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                          cfg=cfg)

    dir = "/opt/wangwei/funsound_onnx/dataset/科学_磁铁"
    Num, Den = test_datadir(am_model,f"{dir}/TR")
    print(Num/Den)
    Num, Den = test_datadir(am_model,f"{dir}/ST")
    print(Num/Den)
    Num, Den = test_datadir(am_model,f"{dir}/MX")
    print(Num/Den)
