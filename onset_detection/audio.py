import IPython.display as ipd
import librosa
import librosa.display
import numpy as np
from madmom.audio import Spectrogram
import matplotlib.pyplot as plt
import torch
import torchaudio
import os
import noisereduce
from scipy.io import wavfile
from pydub import AudioSegment

def read_audio(file_path, trim_interval=None, mono=True, print_it=False):
    audio, sr = librosa.load(file_path, mono=mono)
    audio_dim = len(audio.shape)
    if not mono and audio_dim == 1:
        audio = np.asarray((audio, audio))
    if trim_interval is not None:
        ti = trim_interval[0]
        tf = trim_interval[1]
        audio = trim_audio(audio, sr, ti, tf, mono)
    if print_it:
        print(audio.shape)
        print(sr)
    return audio, sr

def save_audio(filename, sampling_rate, audio_data):
    # Write the audio data to a WAV file
    wavfile.write(filename, sampling_rate, audio_data)
    return


def trim_audio(audio, sr, ti, tf, mono=True):
    i = ti * sr
    f = tf * sr
    if mono:
        t_audio = audio[i:f]
    else:
        t_audio = audio[:, i:f]
    return t_audio


def plot_audio(x, sr):
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(x, sr=sr)
    return


def play_audio(file_path, sr):
    ipd.Audio(file_path, rate=sr)  # load a local WAV file

    # Function that calculates the total seconds of an audio file


def audio_seconds(audio, sr):
    samples = audio.shape[0]
    s = samples / sr
    return s


def show_spectrogram(x, sr, y_scale='linear', frame_size=2048):
    X = librosa.stft(x, n_fft=frame_size)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis=y_scale, n_fft=frame_size, cmap='viridis')
    return


def madmom_spectrogram(file_path):
    spec = Spectrogram(file_path)
    return spec


def rave(file_path, audio, sr, model='percussion', trim=True):
    file_name = file_path.split('/')[-1].split('.')[0]

    folder = f'../results/rave/'
    if not trim:
        folder = folder + 'full_songs/'
    folder = folder + model

    # Check if the folder exists
    if not os.path.exists(folder):
        # If it doesn't exist, create it
        os.makedirs(folder)

    output_file = f'{folder}/{file_name}-{model}.mp3'

    # Load the model
    model = torch.jit.load(f'../models/{model}.ts')
    

    #x, sr = li.load('audio.wav',sr=44100)
    #x = torch.from_numpy(x).reshape(1,1,-1)
    
    x = torch.from_numpy(audio).reshape(1,1,-1)
    #audio_tensor = torch.from_numpy(audio)
    #input_tensor = torch.unsqueeze(torch.unsqueeze(audio_tensor, 0), 0)
    #x = torch.unsqueeze(torch.unsqueeze(audio_tensor, 0), 0)

    # encode and decode the audio with RAVE
    z = model.encode(x)
    x_hat = model.decode(z)
    waveform_tensor = torch.squeeze(x_hat, 0)
    #.detach().numpy().reshape(-1)


    # Apply the pre-trained model and squeeze it
    #with torch.no_grad():
     #   output_tensor = model(input_tensor)
    #waveform_tensor = torch.squeeze(output_tensor, 0)

    # extra_samples = waveform_tensor.shape[1] - audio.shape[0]
    # if extra_samples > 0:
    # Remove initial rhythms
    # waveform_tensor = waveform_tensor[:, extra_samples*36:]

    # RAVE adds extra initial samples that must be removed to synchronize with the original song.
    #offset = 24000
    #waveform_tensor = waveform_tensor[:, offset:]
    # Save the tensor into an audio file and load it
    torchaudio.save(output_file, waveform_tensor, sr)
    return output_file


def same_length(audio1, audio2):
    length = max(len(audio1), len(audio2))
    #print(f'audio1: {len(audio1)}\n')
    #print(f'audio2: {len(audio2)}\n')
    #print(f'max_len: {length}\n')

    audio1 = np.pad(audio1, (0, length - len(audio1)), 'constant')
    audio2 = np.pad(audio2, (0, length - len(audio2)), 'constant')

    #print(f'new_audio1: {len(audio1)}\n')
    #print(f'new_audio2: {len(audio2)}\n')


    return audio1, audio2


def mix_audio(no_drums, version, mixing_factor=15):
    no_drums, version = same_length(no_drums, version)

    return no_drums + version * mixing_factor

def rave_mixing(audio, path, sr, drums_audio, no_drums_audio, song_audio, model='GMDrums_v3_29-09_3M_streaming', no_drums_mixing=30):
    rave_path = rave(path, audio, sr, model=model)

    rave_audio, rave_sr = read_audio(rave_path)
    
    print("Original song")
    display(ipd.Audio(song_audio, rate=sr))

    print('Taps')
    display(ipd.Audio(audio, rate=sr))

    print('RAVE output')
    display(ipd.Audio(rave_audio, rate=rave_sr))
    
    print('Drum track')
    display(ipd.Audio(drums_audio, rate=sr))
    
    print('Reconstructed Drum track')
    rdrums_path = rave(path, drums_audio, sr, model=model)
    rave_drums, rave_drums_sr = read_audio(rdrums_path)
    display(ipd.Audio(rave_drums, rate=rave_drums_sr))
    
    print('Taps + drum track')
    drums_version = mix_audio(drums_audio, audio)
    display(ipd.Audio(drums_version, rate=sr))

    rave_version, lag = synch_signals(audio, rave_audio, mixing_factor=1)

    print('Taps + RAVE')
    display(ipd.Audio(rave_version, rate=rave_sr))
    
    taps_version = mix_audio(no_drums_audio, audio, mixing_factor=no_drums_mixing)
                             
    print('Remixing the taps onto the original song')
    display(ipd.Audio(taps_version, rate=sr))

    rave_version = mix_audio(no_drums_audio, rave_audio, mixing_factor=no_drums_mixing)

    print("Remixing RAVE's output onto the original song")
    display(ipd.Audio(rave_version, rate=rave_sr))
       
    rave_version = mix_audio(song_audio, rave_audio, mixing_factor=no_drums_mixing*2)
    
    print("Remixing RAVE's output onto the original track with drums")
    display(ipd.Audio(rave_version, rate=rave_sr))
    
    print("Remixing RAVE's reconstructed drums onto the original song")
    rave_version = mix_audio(no_drums_audio, rave_drums, mixing_factor=2)
    display(ipd.Audio(rave_version, rate=rave_sr))

    return

def synch_signals(signal1, signal2, mixing_factor=1, lag=0):
    import numpy as np
    import scipy

    if lag == 0:
        # Calcule a correlação cruzada entre os dois sinais
        cross_correlation = scipy.signal.correlate(signal1, signal2, mode='full')

        # Encontre o deslocamento (lag) correspondente ao pico da correlação
        lag = np.argmax(cross_correlation) - len(signal1) + 1
    #print(lag)

    # Ajuste um dos sinais para sincronizá-los
    if lag > 0:
        signal1 = signal1[lag:]
    elif lag < 0:
        signal2 = signal2[-lag:]
    
    # Agora você pode somar os dois sinais para mixá-los
    mixed_signal = mix_audio(signal1, signal2, mixing_factor)

    return mixed_signal, lag

def limit_volume_np(audio, max_volume_dBFS=-20):
    # Calculate the current volume of the audio
    current_volume_dBFS = 10 * np.log10(np.mean(audio ** 2))

    # Calculate the difference in dBFS to the target volume
    volume_difference = current_volume_dBFS - max_volume_dBFS

    # If the audio is too loud, reduce its volume
    if volume_difference > 0:
        scaling_factor = 10 ** (volume_difference / 20)  # Convert dB to linear scaling factor
        limited_audio = audio / scaling_factor
        return limited_audio
    else:
        return audio  # The audio is already within the volume limit

