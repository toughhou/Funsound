import requests
import time
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def handle_request_errors(func):
    def wrapper(*args, **kwargs):
        try:
            response = func(*args, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"访问失败: {e}")
            exit(1)
    return wrapper

class SpeechRecognitionClient:
    def __init__(self, host="http://www.funsound.cn", port=5003):
        self.host = host
        self.port = port

    @handle_request_errors
    def get_worker_status(self):
        """
        获取当前工作者的状态信息。
        """
        return requests.get(f"{self.host}:{self.port}/worker_status")

    @handle_request_errors
    def submit_task(self, audio_file):
        """
        提交音频文件以进行语音识别任务。
        """
        with open(audio_file, 'rb') as f:
            files = {'file': f}
            return requests.post(f"{self.host}:{self.port}/submit", files=files)

    @handle_request_errors
    def get_task_progress(self, task_id):
        """
        获取任务进度。
        """
        return requests.get(f"{self.host}:{self.port}/task_prgs/{task_id}")

    def monitor_task_progress(self, task_id, max_retries=3600):
        """
        轮询任务进度并输出结果。
        """
        retries = 0
        while retries < max_retries:
            progress_response = self.get_task_progress(task_id)
            if progress_response['code'] == 0:
                status = progress_response['content']['status']

                prgs = progress_response['content']['prgs']
                if prgs:
                    logging.info(prgs)

                if status == "SUCCESS":
                    return progress_response['content']['result']
                if status == "FAIL":
                    return []

            else:
                logging.error("无法获取任务进度")
                exit(1)

            retries += 1
            time.sleep(1)  # 等待1秒后再次查询任务进度

        logging.error("超出最大重试次数，任务未完成")
        exit(1)

def main():
    client = SpeechRecognitionClient()
    # audio_file = "funsound/examples/test1.wav"
    audio_file = "/home/ubuntu/funsound_server/cache/2024-08-29+16:44:43-UKbfLRef9f.wav"

    try:
        # 获取工作者状态
        worker_status = client.get_worker_status()
        logging.info(f"Worker Status: {worker_status}")

        # 提交转写任务
        task_response = client.submit_task(audio_file)
        logging.info(f"Task Submission Response: {task_response}")
        
        # 获取转写结果
        if task_response['code'] == 0:
            task_id = task_response['content']
            result = client.monitor_task_progress(task_id)
            logging.info("Transcription Result:")
            for line in result:
                logging.info(f"{line['start']:.2f} - {line['end']:.2f} {line['text']}")
        else:
            logging.error("任务提交失败")
            exit(1)
    except Exception as e:
        logging.error(f"程序运行过程中发生错误: {e}")
        exit(1)

if __name__ == "__main__":
    main()
