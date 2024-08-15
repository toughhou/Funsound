from funasr_onnx import SenseVoiceSmall
from funsound.utils import *



class SenseVoiceSmallPlus(SenseVoiceSmall):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.SR = 16000

    def __call__(self, waveform_list: list, **kwargs):
        language_input = kwargs.get("language", "auto")
        textnorm_input = kwargs.get("textnorm", "woitn")
        language_list, textnorm_list = self.read_tags(language_input, textnorm_input)
        
        # waveform_list = self.load_data(wav_content, self.frontend.opts.frame_opts.samp_freq)
        waveform_nums = len(waveform_list)
        
        assert len(language_list) == 1 or len(language_list) == waveform_nums, \
            "length of parsed language list should be 1 or equal to the number of waveforms"
        assert len(textnorm_list) == 1 or len(textnorm_list) == waveform_nums, \
            "length of parsed textnorm list should be 1 or equal to the number of waveforms"
        
        asr_res = []
        for beg_idx in tqdm(range(0, waveform_nums, self.batch_size),desc='decoding ..'):
            end_idx = min(waveform_nums, beg_idx + self.batch_size)
            feats, feats_len = self.extract_feat(waveform_list[beg_idx:end_idx])
            _language_list = language_list[beg_idx:end_idx]
            _textnorm_list = textnorm_list[beg_idx:end_idx]
            B = feats.shape[0]
            if len(_language_list) == 1 and B != 1:
                _language_list = _language_list * B
            if len(_textnorm_list) == 1 and B != 1:
                _textnorm_list = _textnorm_list * B
            ctc_logits, encoder_out_lens = self.infer(
                feats,
                feats_len,
                np.array(_language_list, dtype=np.int32),
                np.array(_textnorm_list, dtype=np.int32),
            )
            for b in range(feats.shape[0]):
                # back to torch.Tensor
                if isinstance(ctc_logits, np.ndarray):
                    ctc_logits = torch.from_numpy(ctc_logits).float()
                # support batch_size=1 only currently
                x = ctc_logits[b, : encoder_out_lens[b].item(), :]
                yseq = x.argmax(dim=-1)
                yseq = torch.unique_consecutive(yseq, dim=-1)

                mask = yseq != self.blank_id
                token_int = yseq[mask].tolist()

                asr_res.append(self.tokenizer.decode(token_int))

        return asr_res
    
    def remove_bracket_content(self, text):
        # 使用正则表达式去除尖括号及其内部内容
        return re.sub(r'<.*>', '', text)



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

    am_model = SenseVoiceSmallPlus(
        model_dir=asr_model_name,
        cache_dir=cache_dir,
        quantize=asr_model_quantize,
        intra_op_num_threads=asr_model_ncpu,
        batch_size=asr_model_batchsize
    )
    print(f"Success to load the {asr_model_name}")
    return am_model