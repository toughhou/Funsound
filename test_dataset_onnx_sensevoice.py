from funsound.utils import *
from funsound.onnx.SenseVoiceSmall import init_model
from funsound.compute_mer import compute_min_edit_distance

def test_datadir(am_model, data_dir,audio_format = 'wav'):
    audio_list, trans_list = load_dataset(data_dir,audio_format)

    results = am_model(audio_list)
    Num, Den = 1, 1
    for result, trans_data in zip(results,trans_list):
        result = am_model.remove_bracket_content(result)
        med, den = compute_min_edit_distance(trans_data, result,show=False)
        Num += med
        Den += den
    return Num, Den

if __name__ == "__main__":

    am_model = init_model(asr_model_name = "iic/SenseVoiceSmall")

    Num, Den = test_datadir(am_model,"/opt/wangwei/funsound_onnx/audio/test/3/TR")
    print(Num/Den)
    Num, Den = test_datadir(am_model,"/opt/wangwei/funsound_onnx/audio/test/3/ST")
    print(Num/Den)
    Num, Den = test_datadir(am_model,"/opt/wangwei/funsound_onnx/audio/test/3/MX")
    print(Num/Den)
