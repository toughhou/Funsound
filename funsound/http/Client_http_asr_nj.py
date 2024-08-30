import os
import json
import requests
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Value

class SpeechRecognitionClient:
    def __init__(self, host="http://47.95.170.190", port=5001, log_function=None):
        self.host = host
        self.port = port
        self.log_function = log_function or self.default_log_message

    def default_log_message(self, message):
        print(message)  # 默认情况下，将日志打印到控制台

    def get_worker_status(self):
        """
        获取当前工作者的状态信息。
        """
        try:
            response = requests.get(f"{self.host}:{self.port}/worker_status")
            response.raise_for_status()  # 检查请求是否成功
            return response.json()
        except requests.exceptions.RequestException as e:
            self.log_function(f"访问失败: {e}")
            exit(1)

    def submit_task(self, audio_file):
        """
        提交音频文件以进行语音识别任务。
        """
        try:
            with open(audio_file, 'rb') as f:
                files = {'file': f}
                response = requests.post(f"{self.host}:{self.port}/submit", files=files)
            response.raise_for_status()  # 检查请求是否成功
            return response.json()
        except requests.exceptions.RequestException as e:
            self.log_function(f"访问失败: {e}")
            exit(1)

    def get_task_progress(self, task_id):
        """
        获取任务进度。
        """
        try:
            response = requests.get(f"{self.host}:{self.port}/task_prgs/{task_id}")
            response.raise_for_status()  # 检查请求是否成功
            return response.json()
        except requests.exceptions.RequestException as e:
            self.log_function(f"访问失败: {e}")
            exit(1)

    def monitor_task_progress(self, task_id, log_message):
        """
        轮询任务进度并返回结果。
        """
        while True:
            progress_response = self.get_task_progress(task_id)
            if progress_response['code'] == 0:
                status = progress_response['content']['status']
                prgs = progress_response['content']['prgs']
                log_message(prgs)
                if status in ["SUCCESS", "FAIL"]:
                    return progress_response['content']
            else:
                log_message("无法获取任务进度")
                exit(1)

            time.sleep(1)  # 等待1秒后再次查询任务进度

def transcribe_and_save(client, audio_file, output_json_path, log_file, success_counter):
    """
    提交音频转写任务并保存转写结果为JSON文件，同时将日志输出到log文件。
    """
    def log_message(message):
        with open(log_file, 'a', encoding='utf-8') as log:
            log.write(f"{message}\n")
    
    log_message(f"开始处理文件: {audio_file}")
    
    # 提交转写任务
    task_response = client.submit_task(audio_file)
    log_message(f"Task Submission Response for {audio_file}: {task_response}")
    
    if task_response['code'] == 0:
        task_id = task_response['content']
        content = client.monitor_task_progress(task_id, log_message)
        if content['status'] == "SUCCESS":
            result = content['result']
            for line in result:
                line['query'] = ""
                line['answer'] = ""
                line['drop'] = False
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(content['result'], f, ensure_ascii=False, indent=4)
            log_message(f"Transcription saved to {output_json_path}")
            with success_counter.get_lock():
                success_counter.value += 1
        else:
            log_message(f"任务失败: {audio_file}")
    else:
        log_message(f"任务提交失败: {audio_file}")

def process_audio_files(client, audio_files, success_counter):
    """
    使用多线程处理音频文件的转写任务。
    """
    with ThreadPoolExecutor(max_workers=5) as executor:  # 可以根据需要调整max_workers数量
        futures = []
        for audio_file in audio_files:
            output_json_path = os.path.splitext(audio_file)[0] + ".json"
            log_file = os.path.join("log", os.path.basename(audio_file) + ".log")
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            futures.append(executor.submit(transcribe_and_save, client, audio_file, output_json_path, log_file, success_counter))

        # 等待所有线程完成，并处理可能出现的异常
        for future in futures:
            try:
                future.result()
            except Exception as e:
                client.log_function(f"Error occurred during processing: {e}")

def get_all_audio_files(root_folder):
    """
    获取根文件夹下所有音频文件的列表。
    """
    audio_files = []
    for subdir, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.flac')):  # 检查音频文件格式
                audio_files.append(os.path.join(subdir, file))
    return audio_files

def main():
    main_log_file = "process.log"  # 全局日志文件路径
    def log_message(msg):
        print(msg)
        # with open(main_log_file, 'a', encoding='utf-8') as log:
        #     log.write(f"{msg}\n")

    client = SpeechRecognitionClient(log_function=log_message)
    root_folder = r"D:\work\funsound_client\data"  # 设置为包含多个音频文件夹的根目录路径

    # 获取所有音频文件
    audio_files = get_all_audio_files(root_folder)
    
    # 使用多线程处理所有音频文件
    success_counter = Value('i', 0)  # 成功任务计数器
    process_audio_files(client, audio_files, success_counter)

    # 输出总任务数和成功任务数
    total_tasks = len(audio_files)
    successful_tasks = success_counter.value
    log_message(f"Total tasks: {total_tasks}")
    log_message(f"Successful tasks: {successful_tasks}")

if __name__ == "__main__":
    main()
