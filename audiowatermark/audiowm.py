# https://github.com/wavmark/wavmark

# ffmpeg -i example.wav -ac 1 -ar 16000 new.wav

import numpy as np
import soundfile
import torch
import wavmark


# 1.load model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = wavmark.load_model().to(device)

# 2.create 16-bit payload
# payload = np.random.choice([0, 1], size=16)
payload = np.array([1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0])
print("Payload:", payload)

# 3.read host audio
# the audio should be a single-channel 16kHz wav, you can read it using soundfile:
signal, sample_rate = soundfile.read("guitar16kHz.wav")
# Otherwise, you can use the following function to convert the host audio to single-channel 16kHz format:
# from wavmark.utils import file_reader
# signal = file_reader.read_as_single_channel("example.wav", aim_sr=16000)

# 4.encode watermark
watermarked_signal, _ = wavmark.encode_watermark(model, signal, payload, show_progress=True)
# you can save it as a new wav:
# soundfile.write("guitar16kHz-wm.wav", watermarked_signal, 16000)

# 5.decode watermark
payload_decoded, _ = wavmark.decode_watermark(model, watermarked_signal, show_progress=True)
print(payload_decoded)

BER = (payload != payload_decoded).mean() * 100
print("Decode BER:%.1f" % BER)