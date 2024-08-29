from funsound.common.asr import ASR_OFFLINE
from funsound.utils import *
from funsound.funasr.onnx.offline import SeacoParaformer,PuncTransformer


class ASR(ASR_OFFLINE):
    def __init__(self,
                 cfg_file="",
                 log_file="") -> None:
        super().__init__(log_file)
        self.cfg = load_config(cfg_file)
        self.asr_iter_batchsize = self.cfg['ASR']['iter_batchsize']
        
        
        mkfile(self.log_file)
    
    def init_state(self):
        while self.prgs.qsize():self.prgs.get()
        self.am_model = SeacoParaformer.init_model(cfg=self.cfg)
        self.punc_model = PuncTransformer.init_model(cfg=self.cfg)
        self.log("Init State successfully .")


    def inference(self,
                  audio_file,
                  make_sentence_split="punc"):
        
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
            _, timestamps_batch, *p = self.am_model(audio_list_batch)
            timestamps.extend(timestamps_batch)
            self.progress(total=audio_list_len,cur=e,msg='asr')

        # 提取全部token时间戳
        timestamps2 = []
        for i, timestamp in enumerate(timestamps):
            for line in timestamp:
                line[1] += i*window_sencond
                line[2] += i*window_sencond
                timestamps2.append(line)

        # 断句
        if make_sentence_split == "sil":
            sentences = self.make_sentence_by_sil(timestamps2)
        if make_sentence_split == "punc":
            sentences = self.make_sentence_by_punc(timestamps2)
            
        self.finish = True
        return sentences
    
    def make_sentence_by_sil(self,timestamp_list:list):
        result = []
        tmp = {'start':-1,'end':-1,'text':''}
        for line in timestamp_list:
            if line:
                token ,start,end = line
                if token=='<sil>':
                    if tmp['start']>=0:
                        result.append(tmp)
                    tmp = {'start':-1,'end':-1,'text':''}
                    pass
                else:
                    if tmp['start']<0:
                        tmp['start'] = start 
                    tmp['end'] = end
                    tmp['text'] += token 
        return result
    
    def make_sentence_by_punc(self,timestamp_list:list):
        timestamp_list = [item for item in timestamp_list if item[0] != '<sil>']
        sentence_asr = {'tokens':[], 'timestamps':[]}
        sentence_cache = {'tokens': [], 'timestamps': []}
        for item in timestamp_list:
            token, start, end = item
            sentence_asr['tokens'].append(token)
            sentence_asr['timestamps'].append([start, end])
        sentences_punc, sentence_cache = self.punc_model.punc_process(sentence_asr, sentence_cache)
        if sentence_cache['tokens']:
            sentence_cache['tokens'][-1] += '。'
            sentences_punc.append(sentence_cache)

        result = []
        for sentence in sentences_punc:
            text = self.punc_model.remove_spaces_between_chinese(" ".join(sentence['tokens']))
            result.append({'start':sentence['timestamps'][0][0],
                           'end':sentence['timestamps'][-1][-1],
                           'text':text})
        return result
        
        


    
