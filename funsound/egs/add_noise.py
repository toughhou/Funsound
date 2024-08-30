from pydub import AudioSegment
import os
import math
import random

def calculate_rms(audio_segment):
    """计算音频片段的均方根（RMS）值"""
    return audio_segment.rms

def adjust_noise_to_snr(audio_segment, noise_segment, snr_db):
    """根据目标SNR调整噪声的音量"""
    signal_rms = calculate_rms(audio_segment)
    noise_rms = calculate_rms(noise_segment)
    
    # 计算目标噪声RMS
    target_noise_rms = signal_rms / (10 ** (snr_db / 20))
    
    # 计算需要调整的分贝值
    adjustment_db = 20 * math.log10(target_noise_rms / noise_rms)
    
    return noise_segment + adjustment_db

def add_noise_to_audio(noise_path, audio_paths, output_dir, snr_db=20, target_sample_rate=16000):
    # 加载噪声音频
    noise = AudioSegment.from_file(noise_path)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    for audio_path in audio_paths:
        # 加载待测音频
        audio = AudioSegment.from_file(audio_path)
        
        # 随机选择一个起始位置，确保不会超出噪声音频的长度
        start_pos = random.randint(0, len(noise) - len(audio))
        noise_segment = noise[start_pos:start_pos + len(audio)]

        # 根据目标SNR调整噪声音量
        noise_segment = adjust_noise_to_snr(audio, noise_segment, snr_db)

        # 将噪声与待测音频混合
        combined = audio.overlay(noise_segment)

        # 确保合成后的音频为目标采样率
        combined = combined.set_frame_rate(target_sample_rate)

        # 构建输出文件名
        output_file = os.path.join(output_dir, os.path.basename(audio_path))

        # 保存合成后的音频
        combined.export(output_file, format="wav")
        print(f"保存合成音频到 {output_file}")

if __name__ == "__main__":
    noise_path = "dataset/noise.wav"  # 噪声音频文件
    data_dir = "dataset/keywords"
    audio_format = 'wav'
    audio_list = []
    for utt in sorted(os.listdir(data_dir)):
        utt, format = utt.split(".")
        if format != audio_format: continue
        audio_file = f"{data_dir}/{utt}.{audio_format}"
        audio_list += [audio_file]
        
    snr = 20
    output_dir = f"{data_dir}_snr={snr}"  # 合成音频的保存目录

    # 调用函数进行合成
    add_noise_to_audio(noise_path, audio_list, output_dir, snr_db=snr)
