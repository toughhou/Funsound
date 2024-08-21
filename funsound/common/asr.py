

from funsound.utils import *


class ASR_OFFLINE():
    def __init__(self,
                 log_file="") -> None:
        self.log_file = log_file
        self.finish = False
        self.prgs = queue.Queue()
        mkfile(self.log_file)

    def log(self,text ):
        with open(self.log_file, 'a+') as f:
            print(text, file=f)

    def progress(self,total=None,cur=0,msg=""):
        self.prgs.put({'total':total,
                       'cur':cur,
                       'msg':msg})

    def inference(self):
        pass