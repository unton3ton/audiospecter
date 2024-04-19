import numpy
from matplotlib import pyplot, mlab
import scipy.io.wavfile
from collections import defaultdict

SAMPLE_RATE = 8000 # Hz
WINDOW_SIZE = 2048 # размер окна, в котором делается fft
WINDOW_STEP = 512 # шаг окна

def get_wave_data(wave_filename):
    sample_rate, wave_data = scipy.io.wavfile.read(wave_filename)
    # assert sample_rate == SAMPLE_RATE, sample_rate
    if isinstance(wave_data[0], numpy.ndarray): # стерео
        wave_data = wave_data.mean(1)
    return wave_data

def show_specgram(wave_data):
    fig = pyplot.figure()
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    ax.specgram(wave_data,
        NFFT=WINDOW_SIZE, noverlap=WINDOW_SIZE - WINDOW_STEP, Fs=SAMPLE_RATE)
    pyplot.savefig(f'{filename[:-4]}.png')
    pyplot.show()

filename = 'BerlinAmsterdam.wav'

# wave_data = get_wave_data(filename)
# show_specgram(wave_data)


import os, librosa
import matplotlib.pyplot as plt
import librosa.display as ld

data_path ='Heartbeat_sound'
# os.listdir(data_path)
# ['artifact','extrastole', 'murmur', 'normal', 'unlabel']

sr=22050
normal, sr = librosa.load(filename, sr = 22050)
librosa.display.waveshow(normal, sr=sr, color="orange")