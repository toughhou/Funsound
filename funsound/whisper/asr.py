from faster_whisper import WhisperModel
from funsound.common.asr import ASR_OFFLINE
from funsound.utils import *


class ASR(ASR_OFFLINE):
    def __init__(self,
                 model_id="",
                 cfg_file="",
                 log_file="") -> None:
        super().__init__(log_file)
        self.cfg = load_config(cfg_file)
        self.model_id = model_id
        mkfile(self.log_file)
    
    def init_state(self):
        self.model = WhisperModel(self.model_id, device=self.cfg['ASR']['device'])
        self.log("Init State successfully .")

    def inference(self,audio_file):
        
        lines = []
        segments, info = self.model.transcribe(audio_file,
                                               condition_on_previous_text=self.cfg['ASR']['condition_on_previous_text'],
                                               language=self.cfg['ASR']['language'])
        for segment in segments:
            line = {'start':segment.start,
                    'end':segment.end,
                    'text':segment.text,
                    'role':""}
            lines.append(line)
            self.log(line)
        return lines
        

    