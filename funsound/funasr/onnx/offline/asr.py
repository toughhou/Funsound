from funsound.common.asr import ASR_OFFLINE
from funsound.utils import *
from funsound.funasr.onnx.offline.SeacoParaformer import init_model


class ASR(ASR_OFFLINE):
    def __init__(self,
                 model_id="",
                 cfg_file="",
                 log_file="") -> None:
        super().__init__(log_file)
        self.cfg = load_config(cfg_file)
        self.model_id = model_id
        self.asr_iter_batchsize = self.cfg['ASR']['asr_iter_batchsize']
        mkfile(self.log_file)
    
    def init_state(self):
        while self.prgs.qsize():self.prgs.get()
        self.model = init_model(asr_model_name = self.model_id,
                          cfg=self.cfg)
        self.log("Init State successfully .")

    def inference(self,audio_file,make_sentence_split="sil"):
        
        # 加载音频
        window_sencond = 30
        audio_list = read_audio_with_split(audio_file,window_seconds=window_sencond)
        audio_list_len = len(audio_list)
        self.progress(total=audio_list_len,cur=0,msg='read audio')


        # 语音识别
        timestamps = []
        for i in range(0,audio_list_len,self.asr_iter_batchsize):
            s,e = i, min(i+self.asr_iter_batchsize,audio_list_len)
            audio_list_batch = audio_list[s:e]
            _, timestamps_batch, *p = self.model(audio_list_batch)
            timestamps.extend(timestamps_batch)
            self.progress(total=audio_list_len,cur=e,msg='asr')

        timestamps2 = []
        for i, timestamp in enumerate(timestamps):
            for line in timestamp:
                line[1] += i*window_sencond
                line[2] += i*window_sencond
                timestamps2.append(line)

        if make_sentence_split == "sil":
            sentences = self.model.make_sentence_by_sil(timestamps2)
            
        self.finish = True
        return sentences
        

    
