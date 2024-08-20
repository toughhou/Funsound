

from funsound.utils import *


class ASR_OFFLINE():
    def __init__(self,
                 log_file="") -> None:
        self.log_file = log_file
        mkfile(self.log_file)

    def log(self,text ):
        with open(self.log_file, 'a+') as f:
            print(text, file=f)

    def inference(self):
        pass