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
```shell
    用法：
        # 如果提供标注将计算字错率
        python ContextualParaformer.py <asr.yaml> <audio_file> [audio_trans_file]
        python SenseVoice.py <asr.yaml> <audio_file> [audio_trans_file]

    示例：
        python SenseVocie.py asr.yaml funsound_onnx/examples/test1.wav  funsound_onnx/examples/test1.trans
```