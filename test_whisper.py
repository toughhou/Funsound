from faster_whisper import WhisperModel

model_size = "/opt/wangwei/funsound_onnx/funasr_models/keepitsimple/faster-whisper-large-v3"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

segments, info = model.transcribe("/opt/wangwei/funsound_onnx/funsound/examples/test1.wav",condition_on_previous_text=False)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print(segment)
    # print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))