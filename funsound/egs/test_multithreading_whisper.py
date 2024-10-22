from funsound.whisper.asr import ASR 
from funsound.common.executor import Worker, launch, get_worker_status, submit_task, get_task_progress
from funsound.utils import *

def init_engine(id):
    engine = ASR(cfg_file='funsound/conf/whisper.yaml',
                log_file=f'log/whisper-{id}.log')
    engine.init_state()
    return engine

def processor(self,params):
    audio_file = params[0]
    result = self.engine.inference(audio_file)
    return result

Worker.processor = processor


if __name__ == "__main__":

    nj = 3 # 开启3路
    workers = []
    for id in range(nj):
        engine = init_engine(id)
        worker = Worker(wid=id,log_file=f'log/worker-{id}.log')
        worker.load_engine(engine=engine)
        workers.append(worker)
    launch(workers)
    print(get_worker_status(workers))


    audio_file = "funsound/examples/test1.wav"
    task_id = submit_task(workers,params=[audio_file])

    while 1:
        prgs = get_task_progress(task_id)
        print(prgs)
        if prgs['status'] in ["SUCCESS","FAIL"]:
            if prgs['status'] == "SUCCESS":
                for line in  prgs['result']:
                    print(line)
            break
        time.sleep(1)




