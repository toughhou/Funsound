from funasr_onnx import SeacoParaformer
from funsound.utils import *
from funasr_onnx.utils.timestamp_utils import time_stamp_lfr6_onnx

def dtw1(P, v_ids):
    T, V = P.shape  # T: time steps, V: vocab size
    k = len(v_ids)  # k: number of phonemes in the wake word
    
    # Initialize dp array, with dp[i][j] representing the max score sum for j phonemes up to frame i
    dp = np.full((T + 1, k + 1), -np.inf)
    dp[0][0] = 0  # Starting point, zero phonemes, zero frames
    
    # Dynamic programming to fill dp table
    for i in range(1, T + 1):
        for j in range(1, k + 1):
            # Option 1: Select current frame i for phoneme v_ids[j-1]
            if i >= j:  # Ensure we have enough frames for j phonemes
                dp[i][j] = max(dp[i][j], dp[i-1][j-1] + P[i-1][v_ids[j-1]])
            # Option 2: Do not select the current frame, inherit the last best result
            dp[i][j] = max(dp[i][j], dp[i-1][j])
    
    # The maximum score for selecting all k phonemes in T frames
    max_score = dp[T][k]
    return max_score / k

def dtw2(P, wake_word):
    T, V = P.shape
    k = len(wake_word)
    
    # 初始化动态规划表 dp
    dp = np.full((T, k), -np.inf)  # 使用 -inf 初始化 dp 表
    
    # 初始化第一个音素的得分
    dp[0, 0] = P[0, wake_word[0]]
    
    # 计算dp表
    for t in range(1, T):
        dp[t, 0] = dp[t-1, 0] + P[t, wake_word[0]]
        for i in range(1, k):
            dp[t, i] = max(dp[t-1, i], dp[t-1, i-1]) + P[t, wake_word[i]]
    
    # 计算最终得分，最大值出现在最后一个音素的任何时间步
    max_score = np.max(dp[:, k-1])
    return max_score

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
        TIMESTAMPS = []
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

        # 贪心搜索
        for am_scores,valid_token_lens,us_peaks in  zip(AM_SCORES,VALID_TOKEN_LENS,US_PEAKS):
            # token_ids_valid = self.greedy_search(am_scores,valid_token_lens)
            # token_chs = self.converter.ids2tokens(token_ids_valid)
            # text = "".join(token_chs).replace("</s>","")
            text = self.decode_one(am_scores,valid_token_lens)
            RESULTS.append("".join(text))
            timestamp_str, timestamp_raw = time_stamp_lfr6_onnx(us_peaks, copy.copy(text))
            timestamp_list = self.get_timestamp_list(timestamp_str)
            TIMESTAMPS.append(timestamp_list)
        return RESULTS,TIMESTAMPS, AM_SCORES, VALID_TOKEN_LENS, US_ALPHAS, US_PEAKS
    
    def get_timestamp_list(self,timestamp_str):
        timestamp_list_raw = timestamp_str.split(";")
        timestamp_list = []
        for line in timestamp_list_raw:
            if line:
                token ,start,end = line.split()
                start,end = float(start), float(end)
                timestamp_list.append([token,start,end])
        return timestamp_list

    

    def kws(self,waveform_list,WORDS=[],as_hotwords=True):
        """加载词表"""
        WORDS_IDXS = []
        for WORD in WORDS:
            WORD_IDX = self.converter.tokens2ids(list(WORD))
            WORDS_IDXS.append(WORD_IDX)

        """解码"""
        _, AM_SCORES, VALID_TOKEN_LENS, US_ALPHAS, US_PEAKS = self.__call__(waveform_list=waveform_list,
                                                                                  hotwords=" ".join(WORDS) if as_hotwords else "")
        RESULTS = []
        for am_score, valid_token_len in zip(AM_SCORES, VALID_TOKEN_LENS):
            am_score = am_score[:valid_token_len-1]
            result = []
            for WORD, WORD_IDX in zip(WORDS, WORDS_IDXS):
                tgt_score = am_score[:,WORD_IDX]
                _max = np.max(tgt_score,axis=1)
                mean_score = np.mean(_max)
                result.append([WORD,mean_score])
            result.sort(reverse=True,key=lambda x: x[1])
            RESULTS.append(result)
        return RESULTS
    
    def kws_dtw(self,waveform_list,WORDS=[],as_hotwords=True):
        """加载词表"""
        WORDS_IDXS = []
        for WORD in WORDS:
            WORD_IDX = self.converter.tokens2ids(list(WORD))
            WORDS_IDXS.append(WORD_IDX)

        """解码"""
        _, AM_SCORES, VALID_TOKEN_LENS, US_ALPHAS, US_PEAKS = self.__call__(waveform_list=waveform_list,
                                                                                  hotwords=" ".join(WORDS) if as_hotwords else "")
        RESULTS = []
        for am_score, valid_token_len in zip(AM_SCORES, VALID_TOKEN_LENS):
            am_score = am_score[:valid_token_len-1]
            result = []
            for WORD, WORD_IDX in zip(WORDS, WORDS_IDXS):
                mean_score = dtw1(am_score,WORD_IDX)/len(WORDS_IDXS)
                result.append([WORD,mean_score])
            result.sort(reverse=True,key=lambda x: x[1])
            RESULTS.append(result)
        return RESULTS
    



def init_model(cfg ={}):
    cache_dir = cfg['ASR']['cache_dir']
    model_id = cfg['ASR']['model_id']
    model_quantize = cfg['ASR']['model_quantize']
    model_ncpu = cfg['ASR']['model_ncpu']
    model_batchsize = cfg['ASR']['model_batchsize']

    model = SeacoParaformerPlus(
        model_dir=model_id,
        cache_dir=cache_dir,
        quantize=model_quantize,
        intra_op_num_threads=model_ncpu,
        batch_size=model_batchsize
    )
    print(f"Success to load the {model_id}")
    return model

if __name__ == "__main__":

    pass


