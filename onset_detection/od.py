import madmom
import madmom.features.onsets as od
import math
import numpy as np
from scipy.signal import argrelextrema
from mido import MidiFile, MidiTrack, MetaMessage, Message
from midi2txt import bpm2tempo
from midi2txt.settings import midi_drum_map
from midi2txt.txt_to_midi import midi_delta_time, back_from_midi_time
from midi2audio import FluidSynth
from onset_detection.audio import audio_seconds


# High Frequency Content (HFC) OD
def hfc(spec):
    hfc_df = od.high_frequency_content(spec)
    return hfc_df


def cnn_od(audio):
    proc = od.CNNOnsetProcessor()
    df = proc(audio)
    return df


def normalize_df(df):
    m = max(df)
    normalized_df = df / m
    return normalized_df


def thresholding(df, df_bins, p_bins=0.01):
    n_bins = int(df_bins * p_bins) if p_bins > 0 else 1
    binary_function = argrelextrema(df, np.greater, order=n_bins)
    binary_function = binary_function[0]
    return binary_function


def peak_picking(df, t=0.01):
    pp = od.peak_picking(df, t, pre_avg=1, post_avg=1,
                         pre_max=1, post_max=1)
    return pp


def get_velocities(audio, b_df, df_bins):
    samples = [int(i / df_bins * audio.shape[0]) for i in b_df]

    volumes = [audio[s] for s in samples]

    # Min-Max normalization
    min_val = min(volumes)
    max_val = max(volumes)

    abs_data = [abs(val - min_val) for val in volumes]
    range_abs_data = max(abs_data) - min(abs_data)
    if range_abs_data == 0:
        # Handle the case where all data points are equal
        velocities = [max_val] * len(volumes)
    else:
        velocities = [round((abs(val - min_val) / range_abs_data) * 126 + 1) for val in volumes]

    return velocities


def calculate_onset_times(audio, sr, b_df, df_bins):
    s = audio_seconds(audio, sr)
    onset_times = [x / df_bins * s for x in b_df]
    return onset_times


def get_tempo(df):
    hist_bins, hist_delays = madmom.features.tempo.interval_histogram_comb(df,
                                                                           alpha=0.79)  # alpha = 0.79 set as established in the paper
    tempo = madmom.features.tempo.detect_tempo((hist_bins, hist_delays), fps=100)  # fps = 100 set as in madmom docs
    tempo = int(math.floor(tempo[0][0]) / 2)  # get the most likely tempo
    return tempo


def od2midi(file_path, df, onset_times, velocities):
    # This cell uses the onset_times and DF previously calculated and generates a MIDI file from it
    file_name = file_path.split('/')[-1].split('.')[0]
    midi_output_file = "../results/midi/" + file_name + ".mid"

    bpm = get_tempo(df)
    print(f"BPM = {bpm}")
    program_nr = 0
    use_beats = False
    channel_nr = 10

    with MidiFile() as out:
        ppq = 192
        midi_tempo = bpm2tempo(bpm)
        s_per_tick = midi_tempo / 1000.0 / 1000 / ppq

        track = MidiTrack()
        out.tracks.append(track)
        out.type = 0
        out.ticks_per_beat = ppq

        track.append(MetaMessage('set_tempo', tempo=midi_tempo))

        track.append(Message('program_change', program=program_nr, time=0))
        lastTime = 0
        # times.sort(key=lambda tup: tup[0])

        beat_idx = 0
        last_tempo = None
        last_timesig = None
        drum_note = 3  # hi-hats
        for entry, v in zip(onset_times, velocities):

            cur_time = entry
            # print(cur_time)
            deltaTime = midi_delta_time(cur_time - lastTime, s_per_tick)
            # print(deltaTime)
            lastTime = lastTime + back_from_midi_time(deltaTime, s_per_tick)
            # print(lastTime)
            # lastTime = cur_time

            if drum_note in midi_drum_map:
                note = midi_drum_map[drum_note]
                track.append(Message('note_on', note=note, velocity=v,
                                     time=deltaTime, channel=channel_nr - 1))
                # print('event: note: %d, time: %f'% (note, cur_time))
                track.append(Message('note_off', note=note, velocity=100, time=1, channel=channel_nr - 1))

        out.save(midi_output_file)
        # print("Saved MIDI file")

    audio_output_file = "../results/midi_audio/" + file_name + ".flac"

    # using the default sound font in 44100 Hz sample rate
    fs = FluidSynth()
    fs.midi_to_audio(midi_output_file, audio_output_file)
    # print("Saved audio file")

    return audio_output_file, midi_output_file
