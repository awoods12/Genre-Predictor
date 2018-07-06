from scipy import signal
from scipy.io import wavfile
import os
import numpy as np
from pydub import AudioSegment
from PIL import Image

os.environ['Path'] = os.environ['Path'] + '..\\bin'
AudioSegment.converter = '..\\bin\\avconv.exe'

def mp3_to_wav(file, dir, wav_dir):
    filename = file.strip('.mp3')
    wav_path = wav_dir + '\\' + filename + '.wav'

    AudioSegment.from_mp3(dir + '\\' + file).export(wav_path, format='wav')

def wav_to_png(file, wav_dir, png_dir):
    filename = file.strip('.mp3')
    wav_path = wav_dir + '\\' + filename + '.wav'
    png_path = png_dir + '\\' + filename + '.png'

    sample_rate, samples = wavfile.read(wav_path)
    if samples.ndim == 2:
        samples = np.fft.fft(samples)
        samples = (samples[:, 0] + samples[:, 1]) / 2
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

    spectrogram = np.array(spectrogram)
    pillow_image = Image.fromarray(spectrogram).resize((128, 128), Image.ANTIALIAS)
    pillow_image.convert('L').save(png_path)

def convert(rap_input_dir, rap_wav_dir, rap_png_dir, classical_input_dir,
            classical_wav_dir, classical_png_dir):
    for file in os.listdir(rap_input_dir):
        if file.endswith('.mp3') or file.endswith('.MP3'):
            mp3_to_wav(file, rap_input_dir, rap_wav_dir)
            wav_to_png(file, rap_wav_dir, rap_png_dir)

    for file in os.listdir(classical_input_dir):
        if file.endswith('.mp3') or file.endswith('.MP3'):
            mp3_to_wav(file, classical_input_dir, classical_wav_dir)
            wav_to_png(file, classical_wav_dir, classical_png_dir)

train_rap_dir = '..\\train_files\\Rap'
train_classical_dir = '..\\train_files\\Classical'
train_rap_wav_dir = '..\\train_files\\rapwav'
train_rap_png_dir = '..\\train_files\\rappng'
train_classical_wav_dir = '..\\train_files\\classicalwav'
train_classical_png_dir = '..\\train_files\\classicalpng'

convert(train_rap_dir, train_rap_wav_dir, train_rap_png_dir, train_classical_dir,
        train_classical_wav_dir, train_classical_png_dir)

eval_rap_dir = '..\\eval_files\\RapEval'
eval_classical_dir = '..\\eval_files\\ClassicalEval'
eval_rap_wav_dir = '..\\eval_files\\rapwaveval'
eval_rap_png_dir = '..\\eval_files\\rappngeval'
eval_classical_wav_dir = '..\\eval_files\\classicalwaveval'
eval_classical_png_dir = '..\\eval_files\\classicalpngeval'

convert(eval_rap_dir, eval_rap_wav_dir, eval_rap_png_dir, eval_classical_dir,
        eval_classical_wav_dir, eval_classical_png_dir)

