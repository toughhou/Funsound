import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import ast 
import os 

# 加载处理器和模型
processor = AutoProcessor.from_pretrained("/opt/wangwei/Qwen2-Audio/models/qwen/Qwen2-Audio-7B-Instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained("/opt/wangwei/Qwen2-Audio/models/qwen/Qwen2-Audio-7B-Instruct", 
                                                           device_map="auto",
                                                           cache_dir="./models",
                                                           local_files_only=True)


def get_result(audio_file):
    # 定义中文对话
    conversation = [
        {'role': 'system', 'content': '你是一个乐于助人的助手。'}, 
        {"role": "user", "content": [
            {"type": "audio", "audio_url": audio_file},
            {"type": "text", "text": '现在有命令词：[\'前翻页\', \'后翻页\', \'打开黑屏\', \'关闭黑屏\', \'暂停视频\', \'继续播放视频\']， 请判断该音频是唤醒哪个命令词，分析完成后请以下json格式输出给出答案```json\n{\n\'result\': "命令词"\n}'},
        ]}
    ]
    # print(conversation)

    # 将对话内容转换为模型输入格式
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

    # 加载音频数据
    audio_data, _ = librosa.load(audio_file, sr=processor.feature_extractor.sampling_rate)
    audios = [audio_data]

    # 准备模型输入
    inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
    inputs.input_ids = inputs.input_ids.to("cuda")

    # 生成响应
    generate_ids = model.generate(**inputs, max_length=256)
    generate_ids = generate_ids[:, inputs.input_ids.size(1):]

    # 解码并打印响应
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    try:
        result = ast.literal_eval(response)['result']
    except Exception as e:
        result = ""
    return result


if __name__ == "__main__":

    success = 0
    data_dir = "dataset/keywords_snr=10"
    file_list = os.listdir(data_dir)
    for i,file in enumerate(file_list):
        utt = file.split('.')[0]
        ref = utt.split("_")[1]
        audio_file = "%s/%s.wav"%(data_dir,utt)
        hyp = get_result(audio_file)
        success += ref==hyp
        if ref!=hyp:
            print(utt, ref, audio_file)
        print(i,len(file_list))
    print(len(file_list),success, success/len(file_list))
