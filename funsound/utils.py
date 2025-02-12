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
import os 
import shutil
import time 
import threading 
import queue 
import json 
import random
import string
import datetime
import copy 

class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start

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

def read_audio_file(audio_file):
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
    return pcm_data


def read_audio_bytes(audio_bytes):
    ffmpeg_cmd = [
    'ffmpeg',
    '-i', 'pipe:',  
    '-f', 's16le',
    '-acodec', 'pcm_s16le',
    '-ar', '16k',
    '-ac', '1',
    'pipe:' ]
    with subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False) as proc:
        stdout_data, stderr_data = proc.communicate(input=audio_bytes)
    pcm_data = np.frombuffer(stdout_data, dtype=np.int16)
    return pcm_data


def load_dataset(data_dir,audio_format='wav', has_trans = True):
    utt_list, audio_list, trans_list = [], [], []
    for utt in sorted(os.listdir(data_dir)):
        utt, format = utt.split(".")
        if format!=audio_format:continue
        audio_file = f"{data_dir}/{utt}.{audio_format}"
        trans_file = f"{data_dir}/{utt}.txt"
        audio_data = audio_i2f(read_audio_file(audio_file))
        if has_trans:
            with open(trans_file,'rt') as f:
                trans_data = f.read()
                trans_list += [trans_data]
        audio_list += [audio_data]
        utt_list += [utt]
    return utt_list, audio_list, trans_list

def read_audio_with_split(audio_file,sr=16000,window_seconds=30):
    window_size = int(sr*window_seconds)
    audio_data = audio_i2f(read_audio_file(audio_file))
    audio_length = len(audio_data)
    windows = []
    for i in range(0, audio_length, window_size):
        s, e = i, min(i + window_size, audio_length)
        window = audio_data[s:e]
        windows.append(window)
    return windows
        
def mkdir(path, reset=False):
    if os.path.exists(path):
        if reset:
            shutil.rmtree(path)
            print(f"Removed existing directory: {path}")
            os.makedirs(path)
    else:
        os.makedirs(path)
        print(f"Directory created: {path}")
    return path

def mkfile(path):
    with open(path,'wt') as f:
        pass
    print(f"File created: {path}")


def get_utt(path):
    return os.path.basename(path).split('.')[0]


def load_config(cfg_file):
    with open(cfg_file, 'r') as f:
        cfg = yaml.safe_load(f)
    pprint(cfg)
    return cfg

def json_io(file,mode="read",inp = []):
    if mode=='read':
        f = open(file, "rt",encoding="utf-8")
        data = json.load(f)
        f.close()
        return data
    else:
        with open(file,'w') as f:
            json.dump(inp,
                       f,
                       ensure_ascii=False,
                       indent=4)
        return True

def generate_random_string(n):
    letters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(letters) for i in range(n))
    return random_string

def get_current_time():
    # 获取当前时间
    now = datetime.datetime.now()
    # 格式化时间，精确到秒
    formatted_time = now.strftime("%Y-%m-%d+%H:%M:%S")
    return formatted_time