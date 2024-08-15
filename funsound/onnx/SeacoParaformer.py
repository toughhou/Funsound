
from funasr_onnx import SeacoParaformer
from funsound.utils import *

class SeacoParaformerPlus(SeacoParaformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.SR = 16000
        self.SECONDS_PER_FRAME = 0.02
        self.UPSAMPLE_TIMES = 3

    def greedy_search( self,log_probs, valid_token_len):
        token_ids = log_probs.argmax(axis=-1)
        token_ids_valid = token_ids[:valid_token_len]
        return token_ids_valid

    def __call__(self, waveform_list: list, hotwords: str = "", **kwargs) -> list:

        # 加载热词编码
        hotwords, hotwords_length = self.proc_hotword(hotwords)
        [bias_embed] = self.eb_infer(hotwords, hotwords_length)
        bias_embed = bias_embed.transpose(1, 0, 2)
        _ind = np.arange(0, len(hotwords)).tolist()
        bias_embed = bias_embed[_ind, hotwords_length.tolist()]
        bias_embed = np.expand_dims(bias_embed, axis=0)

        # onnx推理
        waveform_nums = len(waveform_list)
        RESULTS = []
        AM_SCORES = []
        VALID_TOKEN_LENS = []
        US_ALPHAS = []
        US_PEAKS = []
        for beg_idx in tqdm(range(0, waveform_nums, self.batch_size),desc="decoding .."):
            end_idx = min(waveform_nums, beg_idx + self.batch_size)
            # 1.计算mel特征
            feats, feats_len = self.extract_feat(waveform_list[beg_idx:end_idx])
            # 2.热词编码同步复制
            bias_embed_ = np.repeat(bias_embed, feats.shape[0], axis=0)
            # 3. 解码
            am_scores, valid_token_lens,us_alphas, us_peaks = self.bb_infer(feats, feats_len, bias_embed_)

            AM_SCORES.extend(am_scores)
            VALID_TOKEN_LENS.extend(valid_token_lens)
            US_ALPHAS.extend(us_alphas)
            US_PEAKS.extend(us_peaks)

        for am_scores,valid_token_lens in  zip(AM_SCORES,VALID_TOKEN_LENS):
            token_ids_valid = self.greedy_search(am_scores,valid_token_lens)
            token_chs = self.converter.ids2tokens(token_ids_valid)
            text = "".join(token_chs).replace("</s>","")
            RESULTS.append(text)
            

        
        return RESULTS, AM_SCORES, VALID_TOKEN_LENS, US_ALPHAS, US_PEAKS
    
    

def init_model(asr_model_name):
    cfg_file = 'asr.yaml'
    with open(cfg_file, 'r') as f:
        cfg = yaml.safe_load(f)
    pprint(cfg)
    cache_dir = cfg['ASR']['cache_dir']
    asr_model_quantize = cfg['ASR']['asr_model_quantize']
    asr_model_ncpu = cfg['ASR']['asr_model_ncpu']
    asr_model_batchsize = cfg['ASR']['asr_model_batchsize']
    hotwords = cfg['ASR']['hotwords']

    am_model = SeacoParaformerPlus(
        model_dir=asr_model_name,
        cache_dir=cache_dir,
        quantize=asr_model_quantize,
        intra_op_num_threads=asr_model_ncpu,
        batch_size=asr_model_batchsize
    )
    print(f"Success to load the {asr_model_name}")
    return am_model

if __name__ == "__main__":

    pass


