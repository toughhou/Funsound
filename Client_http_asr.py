import requests
import time

HOST = "http://127.0.0.1"
PORT = 5001

def get_worker_status():
    """
    获取当前工作者的状态信息。
    """
    try:
        response = requests.get(f"{HOST}:{PORT}/worker_status")
        response.raise_for_status()  # 检查请求是否成功
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"访问失败: {e}")
        exit(1)

def submit_task(audio_file):
    """
    提交音频文件以进行语音识别任务。
    """
    try:
        with open(audio_file, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{HOST}:{PORT}/submit", files=files)
        response.raise_for_status()  # 检查请求是否成功
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"访问失败: {e}")
        exit(1)

def get_task_progress(task_id):
    """
    获取任务进度。
    """
    try:
        response = requests.get(f"{HOST}:{PORT}/task_prgs/{task_id}")
        response.raise_for_status()  # 检查请求是否成功
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"访问失败: {e}")
        exit(1)

def main():
    audio_file = "funsound/examples/test1.wav"

    # 获取工作者状态
    worker_status = get_worker_status()
    print("Worker Status:", worker_status)

    # 提交转写任务
    task_response = submit_task(audio_file)
    print("Task Submission Response:", task_response)
    
    if task_response['code'] == 0:
        task_id = task_response['content']
    else:
        print("任务提交失败")
        exit(1)

    # 轮询任务进度
    while True:
        progress_response = get_task_progress(task_id)
        print("Task Progress Response:", progress_response)

        if progress_response['code'] == 0:
            status = progress_response['content']['status']
            if status in ["SUCCESS", "FAIL"]:
                if status == "SUCCESS":
                    print("Transcription Result:")
                    for line in progress_response['content']['result']:
                        print(line)
                else:
                    print("任务失败")
                break
        else:
            print("无法获取任务进度")
            exit(1)

        time.sleep(1)  # 等待1秒后再次查询任务进度

if __name__ == "__main__":
    main()
