from funsound.utils import *
TASK_STRUCT = {'status': "INIT", 'prgs':None, 'result': []}

TASKS = {}
TASKS_LOCK = threading.Lock()

class Worker(threading.Thread):
    def __init__(self, wid, log_file):
        super().__init__()
        self.daemon = True
        self.wid = wid
        self.log_file = log_file
        self.queue = queue.Queue()
        self.is_working = threading.Event()
        self._stop_event = threading.Event()
        mkfile(self.log_file)

    def stop(self):
        self._stop_event.set()

    def log(self, text, level="info"):
        with open(self.log_file, 'a+') as f:
            log_message = f"[{level.upper()}] {text}"
            print(log_message, file=f)
    
    def load_engine(self, engine):
        self.engine = engine

    def processor(self, params):
        # This method should be implemented by subclasses to perform specific tasks
        pass

    def handle_task(self, task_id, params):
        TASKS[task_id]['status'] = "PROCESS"
        TASKS[task_id]['prgs'] = self.engine.prgs
        try:
            result = self.processor(params)
            with TASKS_LOCK:
                TASKS[task_id]['status'] = "SUCCESS"
                TASKS[task_id]['result'] = result
            self.log(f"Task {task_id} completed successfully.")
        except Exception as e:
            self.log(f"Task {task_id} failed with error: {str(e)}", level="error")
            with TASKS_LOCK:
                TASKS[task_id]['status'] = "FAIL"
                TASKS[task_id]['result'] = str(e)

    def run(self):
        while not self._stop_event.is_set():
            time.sleep(0.1)
            if not self.queue.empty():
                task_id, *params = self.queue.get()
                print(f"WORKER-{self.wid}: Start task {task_id}.")
                self.is_working.set()
                self.handle_task(task_id, params) # 阻塞
                self.is_working.clear()
                print(f"WORKER-{self.wid}: Done with task {task_id}.")


def launch(workers):
    for worker in workers:
        worker.start()
        print(f"launch the worker:{worker.wid}")

def kill(workers):
    for worker in workers:
        worker.stop()
        print(f"kill the worker:{worker.wid}")

def wait(workers):
    for worker in workers:
        worker.join()
        print(f"wait the worker:{worker.wid}")

def get_worker_status(workers: list):
    return [worker.queue.qsize() + worker.is_working.is_set() for worker in workers]

def get_relax_worker(workers: list):
    # Find the worker with the fewest tasks and not currently working
    min_task_worker = min(workers, key=lambda worker: worker.queue.qsize() + worker.is_working.is_set())
    return min_task_worker

def get_task_progress(task_id):
    with TASKS_LOCK:
        if task_id in TASKS:
            res = {}
            state = TASKS[task_id]
            res['status'] = state['status']
            res['result'] = state['result']
            res['prgs'] = None
            if state['prgs'] and  state['prgs'].qsize():
                while state['prgs'].qsize():
                    tmp = state['prgs'].get()
                res['prgs'] = tmp
            return res
        else:
            return None

    
def submit_task(workers, params:list):
    task_id = generate_random_string(20)
    worker = get_relax_worker(workers)
    worker.queue.put([task_id]+params)
    TASKS[task_id] = TASK_STRUCT.copy()
    return task_id
    

    

