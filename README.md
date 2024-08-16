# Funsound: 基于Funasr的 语音识别简易开发调用接口

## 网站应用
> 离线语音转写：www.funsound.cn

> 联系邮箱: 605686962@qq.com

## 安装
```shell
    pip install -U funasr
    pip install -U funasr-onnx
    pip install -U modelscope
```

## 用法

### 1. 离线语音识别
```python
from funsound.utils import *
from funsound.onnx.offline.SeacoParaformer import init_model
from funsound.compute_mer import compute_mer_text

if __name__ == "__main__":

    # 加载模型
    cfg = load_config('conf/onnx.yaml')
    am_model = init_model(asr_model_name = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                          cfg=cfg)

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
```

### 2. 离线语音唤醒
```python
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
```