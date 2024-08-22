
from funasr_onnx import SeacoParaformer
from funsound.utils import *


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
    



def init_model(asr_model_name,cfg ={}):
    cache_dir = cfg['ASR']['cache_dir']
    asr_model_quantize = cfg['ASR']['asr_model_quantize']
    asr_model_ncpu = cfg['ASR']['asr_model_ncpu']
    asr_model_batchsize = cfg['ASR']['asr_model_batchsize']

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


