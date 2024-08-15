import numpy as np 
import subprocess
import wave 
import heapq
import yaml 
import sys 
from pprint import pprint
from tqdm import tqdm
import torch
import  re
from scipy.ndimage import uniform_filter1d
import os 
import shutil

def audio_f2i(data, width=16):
    """将浮点数音频数据转换为整数音频数据。"""
    data = np.array(data)
    return np.int16(data * (2 ** (width - 1)))

def audio_i2f(data, width=16):
    """将整数音频数据转换为浮点数音频数据。"""
    data = np.array(data)
    return np.float32(data / (2 ** (width - 1)))

def save_wavfile(path, wave_data):
    """保存音频数据为wav文件。"""
    with wave.open(path, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(wave_data.tobytes())
    print(f"Successfully saved wavfile: {path} ..")

def read_audio_data(audio_file):
    """读取音频文件数据并转换为PCM格式。"""
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', audio_file,
        '-f', 's16le',
        '-acodec', 'pcm_s16le',
        '-ar', '16k',
        '-ac', '1',
        'pipe:']
    with subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False) as proc:
        stdout_data, stderr_data = proc.communicate()
    pcm_data = np.frombuffer(stdout_data, dtype=np.int16)
    pcm_data = audio_i2f(pcm_data)
    return pcm_data

def sliding_window(audio_data, sample_rate=16000, window_size_sec=0.5):
    win_size = int(window_size_sec * sample_rate)
    rms = np.sqrt(uniform_filter1d(audio_data**2, size=win_size, mode='reflect'))
    return rms, audio_data / np.exp(rms*20)


def load_dataset(data_dir,audio_format='wav'):
    audio_list, trans_list = [], []
    for utt in sorted(os.listdir(data_dir)):
        utt, format = utt.split(".")
        if format!=audio_format:continue
        audio_file = f"{data_dir}/{utt}.{audio_format}"
        trans_file = f"{data_dir}/{utt}.txt"
        audio_data = read_audio_data(audio_file)
        with open(trans_file,'rt') as f:
            trans_data = f.read()
        audio_list += [audio_data]
        trans_list += [trans_data]
    return audio_list, trans_list

def read_audio_with_split(audio_file,sr=16000,window_seconds=30):
    window_size = int(sr*window_seconds)
    audio_data = read_audio_data(audio_file)
    audio_length = len(audio_data)
    windows = []
    for i in range(0, audio_length, window_size):
        s, e = i, min(i + window_size, audio_length)
        window = audio_data[s:e]
        windows.append(window)
    return windows
        
def mkdir(path, reset=False):
    # 检查目录是否存在
    if os.path.exists(path):
        if reset:
            # 安全地删除目录及其所有内容
            shutil.rmtree(path)
            print(f"Removed existing directory: {path}")

    # 无论是否需要重置，都确保目录被创建
    try:
        os.makedirs(path)
        print(f"Directory created: {path}")
    except OSError as e:
        # 捕捉可能的异常，如权限不足或路径为文件时无法创建目录
        print(f"Error creating directory: {e}")
        return None  # 或者根据你的错误处理策略抛出异常或返回特定值

    return path
