import os
import json
import requests
import time
import threading
from concurrent.futures import ThreadPoolExecutor

class SpeechRecognitionClient:
    def __init__(self, host="http://47.95.170.190", port=5001):
        self.host = host
        self.port = port

    def get_worker_status(self):
        """
        获取当前工作者的状态信息。
        """
        try:
            response = requests.get(f"{self.host}:{self.port}/worker_status")
            response.raise_for_status()  # 检查请求是否成功
            return response.json()
        except requests.exceptions.RequestException as e:
            self.log_message(f"访问失败: {e}")
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
            self.log_message(f"访问失败: {e}")
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
            self.log_message(f"访问失败: {e}")
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

def transcribe_and_save(client, audio_file, output_json_path, log_file):
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
        result = client.monitor_task_progress(task_id, log_message)
        if result['status'] == "SUCCESS":
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(result['result'], f, ensure_ascii=False, indent=4)
            log_message(f"Transcription saved to {output_json_path}")
        else:
            log_message(f"任务失败: {audio_file}")
    else:
        log_message(f"任务提交失败: {audio_file}")

def process_audio_files(client, audio_files):
    """
    使用多线程处理音频文件的转写任务。
    """
    with ThreadPoolExecutor(max_workers=5) as executor:  # 可以根据需要调整max_workers数量
        futures = []
        for audio_file in audio_files:
            output_json_path = os.path.splitext(audio_file)[0] + ".json"
            log_file = os.path.join("log", os.path.basename(audio_file) + ".log")
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            futures.append(executor.submit(transcribe_and_save, client, audio_file, output_json_path, log_file))

        # 等待所有线程完成
        for future in futures:
            future.result()

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
    client = SpeechRecognitionClient()
    root_folder = "path/to/audio/folders"  # 设置为包含多个音频文件夹的根目录路径

    # 获取所有音频文件
    audio_files = get_all_audio_files(root_folder)

    # 使用多线程处理所有音频文件
    process_audio_files(client, audio_files)

if __name__ == "__main__":
    main()
