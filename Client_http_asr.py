import requests
import time

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
            print(f"访问失败: {e}")
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
            print(f"访问失败: {e}")
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
            print(f"访问失败: {e}")
            exit(1)

    def monitor_task_progress(self, task_id):
        """
        轮询任务进度并输出结果。
        """
        while True:
            progress_response = self.get_task_progress(task_id)
            if progress_response['code'] == 0:
                status = progress_response['content']['status']

                prgs = progress_response['content']['prgs']
                if prgs:
                    print(prgs)

                if status == "SUCCESS":
                    return progress_response['content']['result']
                if status == "FAIL":
                    return []

            else:
                print("无法获取任务进度")
                exit(1)

            time.sleep(1)  # 等待1秒后再次查询任务进度

def main():
    client = SpeechRecognitionClient()
    audio_file = r"D:\work\funsound_client\data\语文_小马过河.mp3"

    # 获取工作者状态
    worker_status = client.get_worker_status()
    print("Worker Status:", worker_status)

    # 提交转写任务
    task_response = client.submit_task(audio_file)
    print("Task Submission Response:", task_response)
    
    # 获取转写结果
    if task_response['code'] == 0:
        task_id = task_response['content']
        result = client.monitor_task_progress(task_id)
        print("Transcription Result:")
        for line in result:
            print(f"%.2f - %.2f %s"%(line['start'], line['end'], line['text']))
    else:
        print("任务提交失败")
        exit(1)

if __name__ == "__main__":
    main()
