# GrooVAE functions.
# Code taken from https://colab.research.google.com/github/tensorflow/magenta-demos/blob/master/colab-notebooks/GrooVAE.ipynb#scrollTo=I47LxktGbwMF


import tensorflow_datasets as tfds
import tensorflow as tf

# Allow python to pick up the newly-installed fluidsynth lib.
# This is only needed for the hosted Colab environment.
# import ctypes.util
# orig_ctypes_util_find_library = ctypes.util.find_library
# def proxy_find_library(lib):
#  if lib == 'fluidsynth':
#    return 'libfluidsynth.so.1'
#  else:
#    return orig_ctypes_util_find_library(lib)
# ctypes.util.find_library = proxy_find_library

#print('Importing software libraries...')

import copy, warnings, librosa, numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Colab/Notebook specific stuff
import IPython.display
from IPython.display import Audio
# from google.colab import files

# Magenta specific stuff
from magenta.models.music_vae import configs
from magenta.models.music_vae.trained_model import TrainedModel
from magenta.models.music_vae import data
import note_seq
from note_seq import midi_synth
from note_seq.sequences_lib import concatenate_sequences
from note_seq.protobuf import music_pb2


# Define some functions

# If a sequence has notes at time before 0.0, scootch them up to 0
def start_notes_at_0(s):
    for n in s.notes:
        if n.start_time < 0:
            n.end_time -= n.start_time
            n.start_time = 0
    return s


def play(note_sequence, sf2_path='Standard_Drum_Kit.sf2'):
    if sf2_path:
        audio_seq = midi_synth.fluidsynth(start_notes_at_0(note_sequence), sample_rate=44100, sf2_path=sf2_path)
        IPython.display.display(IPython.display.Audio(audio_seq, rate=44100))
    else:
        note_seq.play_sequence(start_notes_at_0(note_sequence), synth=note_seq.fluidsynth)


# Some midi files come by default from different instrument channels
# Quick and dirty way to set midi files to be recognized as drums
def set_to_drums(ns):
    for n in ns.notes:
        n.instrument = 9
        n.is_drum = True


def unset_to_drums(ns):
    for note in ns.notes:
        note.is_drum = False
        note.instrument = 0
    return ns


# quickly change the tempo of a midi sequence and adjust all notes
def change_tempo(note_sequence, new_tempo):
    new_sequence = copy.deepcopy(note_sequence)
    ratio = note_sequence.tempos[0].qpm / new_tempo
    for note in new_sequence.notes:
        note.start_time = note.start_time * ratio
        note.end_time = note.end_time * ratio
    new_sequence.tempos[0].qpm = new_tempo
    return new_sequence


# def download(note_sequence, filename):
#     note_seq.sequence_proto_to_midi_file(note_sequence, filename)
#     files.download(filename)


# def download_audio(audio_sequence, filename, sr):
#     librosa.output.write_wav(filename, audio_sequence, sr=sr, norm=True)
#     files.download(filename)


# Load some configs to be used later
dc_quantize = configs.CONFIG_MAP['groovae_2bar_humanize'].data_converter
dc_tap = configs.CONFIG_MAP['groovae_2bar_tap_fixed_velocity'].data_converter
dc_hihat = configs.CONFIG_MAP['groovae_2bar_add_closed_hh'].data_converter
dc_4bar = configs.CONFIG_MAP['groovae_4bar'].data_converter


# quick method for removing microtiming and velocity from a sequence
def get_quantized_2bar(s, velocity=0):
    new_s = dc_quantize.from_tensors(dc_quantize.to_tensors(s).inputs)[0]
    new_s = change_tempo(new_s, s.tempos[0].qpm)
    if velocity != 0:
        for n in new_s.notes:
            n.velocity = velocity
    return new_s


# quick method for turning a drumbeat into a tapped rhythm
def get_tapped_2bar(s, velocity=85, ride=False):
    new_s = dc_tap.from_tensors(dc_tap.to_tensors(s).inputs)[0]
    new_s = change_tempo(new_s, s.tempos[0].qpm)
    if velocity != 0:
        for n in new_s.notes:
            n.velocity = velocity
    if ride:
        for n in new_s.notes:
            n.pitch = 42
    return new_s


# quick method for removing hi-hats from a sequence
def get_hh_2bar(s):
    new_s = dc_hihat.from_tensors(dc_hihat.to_tensors(s).inputs)[0]
    new_s = change_tempo(new_s, s.tempos[0].qpm)
    return new_s


# Calculate quantization steps but do not remove microtiming
def quantize(s, steps_per_quarter=4):
    return note_seq.sequences_lib.quantize_note_sequence(s, steps_per_quarter)


# Destructively quantize a midi sequence
def flatten_quantization(s):
    beat_length = 60. / s.tempos[0].qpm
    step_length = beat_length / 4  # s.quantization_info.steps_per_quarter
    new_s = copy.deepcopy(s)
    for note in new_s.notes:
        note.start_time = step_length * note.quantized_start_step
        note.end_time = step_length * note.quantized_end_step
    return new_s


# Calculate how far off the beat a note is
def get_offset(s, note_index):
    q_s = flatten_quantization(quantize(s))
    true_onset = s.notes[note_index].start_time
    quantized_onset = q_s.notes[note_index].start_time
    diff = quantized_onset - true_onset
    beat_length = 60. / s.tempos[0].qpm
    step_length = beat_length / 4  # q_s.quantization_info.steps_per_quarter
    offset = diff / step_length
    return offset


def is_4_4(s):
    ts = s.time_signatures[0]
    return (ts.numerator == 4 and ts.denominator == 4)


def preprocess_4bar(s):
    return dc_4bar.from_tensors(dc_4bar.to_tensors(s).outputs)[0]


def preprocess_2bar(s):
    return dc_quantize.from_tensors(dc_quantize.to_tensors(s).outputs)[0]


def _slerp(p0, p1, t):
    """Spherical linear interpolation."""
    omega = np.arccos(np.dot(np.squeeze(p0 / np.linalg.norm(p0)),
                             np.squeeze(p1 / np.linalg.norm(p1))))
    so = np.sin(omega)
    return np.sin((1.0 - t) * omega) / so * p0 + np.sin(t * omega) / so * p1


# print('Downloading drum samples...')
# Download a drum kit for playing drum midi
# !gsutil - q - m
# cp
# gs: // magentadata / soundfonts / Standard_Drum_Kit.sf2.

# print("Download MIDI data...")

# Load MIDI files from GMD with MIDI only (no audio) as a tf.data.Dataset
dataset_2bar = tfds.as_numpy(tfds.load(
    name="groove/2bar-midionly",
    split=tfds.Split.VALIDATION,
    try_gcs=True))

dev_sequences = [quantize(note_seq.midi_to_note_sequence(features["midi"])) for features in dataset_2bar]
_ = [set_to_drums(s) for s in dev_sequences]
dev_sequences = [s for s in dev_sequences if is_4_4(s) and len(s.notes) > 0 and s.notes[
    -1].quantized_end_step > note_seq.steps_per_bar_in_quantized_sequence(s)]

dataset_4bar = tfds.as_numpy(tfds.load(
    name="groove/4bar-midionly",
    split=tfds.Split.VALIDATION,
    try_gcs=True))

dev_sequences_4bar = [quantize(note_seq.midi_to_note_sequence(features["midi"])) for features in dataset_4bar]
_ = [set_to_drums(s) for s in dev_sequences_4bar]
dev_sequences_4bar = [s for s in dev_sequences_4bar if is_4_4(s) and len(s.notes) > 0 and s.notes[
    -1].quantized_end_step > note_seq.steps_per_bar_in_quantized_sequence(s)]

# print("Loading model checkpoints...")

# Download all the models
# !gsutil - q - m
# cp
# gs: // magentadata / models / music_vae / checkpoints / groovae_ *.tar.
GROOVAE_4BAR = "groovae_4bar.tar"
GROOVAE_2BAR_HUMANIZE = "groovae_2bar_humanize.tar"
GROOVAE_2BAR_HUMANIZE_NOKL = "groovae_2bar_humanize_nokl.tar"
GROOVAE_2BAR_HITS_CONTROL = "groovae_2bar_hits_control.tar"
GROOVAE_2BAR_TAP_FIXED_VELOCITY = "groovae_2bar_tap_fixed_velocity.tar"
GROOVAE_2BAR_ADD_CLOSED_HH = "groovae_2bar_add_closed_hh.tar"
GROOVAE_2BAR_HITS_CONTROL_NOKL = "groovae_2bar_hits_control_nokl.tar"

# print("Downloading audio data...")
# !gsutil - q - m
# cp
# gs: // magentadata / models / music_vae / groovae_colab / * wav.

# @title Load checkpoint

config_2bar_tap = configs.CONFIG_MAP['groovae_2bar_tap_fixed_velocity']
groovae_2bar_tap = TrainedModel(config_2bar_tap, 1, checkpoint_dir_or_path=GROOVAE_2BAR_TAP_FIXED_VELOCITY)


def mix_tracks(y1, y2, stereo=False):
    l = max(len(y1), len(y2))
    y1 = librosa.util.fix_length(y1, l)
    y2 = librosa.util.fix_length(y2, l)

    if stereo:
        return np.vstack([y1, y2])
    else:
        return y1 + y2


def make_click_track(s):
    last_note_time = max([n.start_time for n in s.notes])
    beat_length = 60. / s.tempos[0].qpm
    i = 0
    times = []
    while i * beat_length < last_note_time:
        times.append(i * beat_length)
        i += 1
    return librosa.clicks(times)


def drumify(s, model, temperature=1.0):
    encoding, mu, sigma = model.encode([s])
    decoded = model.decode(encoding, length=32, temperature=temperature)
    return decoded[0]


def combine_sequences(seqs):
    # assumes a list of 2 bar seqs with constant tempo
    for i, seq in enumerate(seqs):
        shift_amount = i * (60 / seqs[0].tempos[0].qpm * 4 * 2)
        if shift_amount > 0:
            seqs[i] = note_seq.sequences_lib.shift_sequence_times(seq, shift_amount)
    return note_seq.sequences_lib.concatenate_sequences(seqs)


def combine_sequences_with_lengths(sequences, lengths):
    seqs = copy.deepcopy(sequences)
    total_shift_amount = 0
    for i, seq in enumerate(seqs):
        if i == 0:
            shift_amount = 0
        else:
            shift_amount = lengths[i - 1]
        total_shift_amount += shift_amount
        if total_shift_amount > 0:
            seqs[i] = note_seq.sequences_lib.shift_sequence_times(seq, total_shift_amount)
    combined_seq = music_pb2.NoteSequence()
    for i in range(len(seqs)):
        tempo = combined_seq.tempos.add()
        tempo.qpm = seqs[i].tempos[0].qpm
        tempo.time = sum(lengths[0:i - 1])
        for note in seqs[i].notes:
            combined_seq.notes.extend([copy.deepcopy(note)])
    return combined_seq


def get_audio_start_time(y, sr):
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    onset_times = librosa.onset.onset_detect(y, sr, units='time')
    start_time = onset_times[0]
    return start_time


def audio_tap_to_note_sequence(f, velocity_threshold=30):
    y, sr = librosa.load(f)
    # pad the beginning to avoid errors with onsets right at the start
    y = np.concatenate([np.zeros(1000), y])
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    # try to guess reasonable tempo
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    onset_frames = librosa.onset.onset_detect(y, sr, units='frames')
    onset_times = librosa.onset.onset_detect(y, sr, units='time')
    start_time = onset_times[0]
    onset_strengths = librosa.onset.onset_strength(y, sr)[onset_frames]
    normalized_onset_strengths = onset_strengths / np.max(onset_strengths)
    onset_velocities = np.int32(normalized_onset_strengths * 127)
    note_sequence = music_pb2.NoteSequence()
    note_sequence.tempos.add(qpm=tempo)
    for onset_vel, onset_time in zip(onset_velocities, onset_times):
        if onset_vel > velocity_threshold and onset_time >= start_time:  # filter quietest notes
            note_sequence.notes.add(
                instrument=9, pitch=42, is_drum=True,
                velocity=onset_vel,  # use fixed velocity here to avoid overfitting
                start_time=onset_time - start_time,
                end_time=onset_time - start_time)

    return note_sequence


# Allow encoding of a sequence that has no extracted examples
# by adding a quiet note after the desired length of time
def add_silent_note(note_sequence, num_bars):
    tempo = note_sequence.tempos[0].qpm
    length = 60 / tempo * 4 * num_bars
    note_sequence.notes.add(
        instrument=9, pitch=42, velocity=0, start_time=length - 0.02,
        end_time=length - 0.01, is_drum=True)


def get_bar_length(note_sequence):
    tempo = note_sequence.tempos[0].qpm
    return 60 / tempo * 4


def sequence_is_shorter_than_full(note_sequence):
    return note_sequence.notes[-1].start_time < get_bar_length(note_sequence)


def get_rhythm_elements(y, sr):
    onset_env = librosa.onset.onset_strength(y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, max_tempo=180)[0]
    onset_times = librosa.onset.onset_detect(y, sr, units='time')
    onset_frames = librosa.onset.onset_detect(y, sr, units='frames')
    onset_strengths = librosa.onset.onset_strength(y, sr)[onset_frames]
    normalized_onset_strengths = onset_strengths / np.max(onset_strengths)
    onset_velocities = np.int32(normalized_onset_strengths * 127)

    return tempo, onset_times, onset_frames, onset_velocities


def make_tap_sequence(tempo, onset_times, onset_frames, onset_velocities,
                      velocity_threshold, start_time, end_time):
    note_sequence = music_pb2.NoteSequence()
    note_sequence.tempos.add(qpm=tempo)
    for onset_vel, onset_time in zip(onset_velocities, onset_times):
        if onset_vel > velocity_threshold and onset_time >= start_time and onset_time < end_time:  # filter quietest notes
            note_sequence.notes.add(
                instrument=9, pitch=42, is_drum=True,
                velocity=onset_vel,  # model will use fixed velocity here
                start_time=onset_time - start_time,
                end_time=onset_time - start_time + 0.01
            )
    return note_sequence


def audio_to_drum(f, velocity_threshold=30, temperature=1., force_sync=False, start_windows_on_downbeat=False):
    y, sr = librosa.load(f)
    # pad the beginning to avoid errors with onsets right at the start
    y = np.concatenate([np.zeros(1000), y])

    clip_length = float(len(y)) / sr

    tap_sequences = []
    # Loop through the file, grabbing 2-bar sections at a time, estimating
    # tempos along the way to try to handle tempo variations

    tempo, onset_times, onset_frames, onset_velocities = get_rhythm_elements(y, sr)

    initial_start_time = onset_times[0]

    start_time = onset_times[0]
    beat_length = 60 / tempo
    two_bar_length = beat_length * 8
    end_time = start_time + two_bar_length

    start_times = []
    lengths = []
    tempos = []

    start_times.append(start_time)
    lengths.append(end_time - start_time)
    tempos.append(tempo)

    tap_sequences.append(make_tap_sequence(tempo, onset_times, onset_frames,
                                           onset_velocities, velocity_threshold, start_time, end_time))

    start_time += two_bar_length;
    end_time += two_bar_length

    while start_time < clip_length:
        start_sample = int(librosa.core.time_to_samples(start_time, sr=sr))
        end_sample = int(librosa.core.time_to_samples(start_time + two_bar_length, sr=sr))
        current_section = y[start_sample:end_sample]
        tempo = librosa.beat.tempo(onset_envelope=librosa.onset.onset_strength(current_section, sr=sr), max_tempo=180)[
            0]

        beat_length = 60 / tempo
        two_bar_length = beat_length * 8

        end_time = start_time + two_bar_length

        start_times.append(start_time)
        lengths.append(end_time - start_time)
        tempos.append(tempo)

        tap_sequences.append(make_tap_sequence(tempo, onset_times, onset_frames,
                                               onset_velocities, velocity_threshold, start_time, end_time))

        start_time += two_bar_length;
        end_time += two_bar_length

    # if there's a long gap before the first note, back it up close to 0
    def _shift_notes_to_beginning(s):
        start_time = s.notes[0].start_time
        if start_time > 0.1:
            for n in s.notes:
                n.start_time -= start_time
                n.end_time -= start_time
        return start_time

    def _shift_notes_later(s, start_time):
        for n in s.notes:
            n.start_time += start_time
            n.end_time += start_time

    def _sync_notes_with_onsets(s, onset_times):
        for n in s.notes:
            n_length = n.end_time - n.start_time
            closest_onset_index = np.argmin(np.abs(n.start_time - onset_times))
            n.start_time = onset_times[closest_onset_index]
            n.end_time = n.start_time + n_length

    drum_seqs = []
    for s in tap_sequences:
        try:
            if sequence_is_shorter_than_full(s):
                add_silent_note(s, 2)

            if start_windows_on_downbeat:
                note_start_time = _shift_notes_to_beginning(s)
            h = drumify(s, groovae_2bar_tap, temperature=temperature)
            h = change_tempo(h, s.tempos[0].qpm)

            if start_windows_on_downbeat and note_start_time > 0.1:
                _shift_notes_later(s, note_start_time)

            drum_seqs.append(h)
        except:
            continue

    combined_tap_sequence = start_notes_at_0(combine_sequences_with_lengths(tap_sequences, lengths))
    combined_drum_sequence = start_notes_at_0(combine_sequences_with_lengths(drum_seqs, lengths))

    if force_sync:
        _sync_notes_with_onsets(combined_tap_sequence, onset_times)
        _sync_notes_with_onsets(combined_drum_sequence, onset_times)

    full_tap_audio = librosa.util.normalize(midi_synth.fluidsynth(combined_tap_sequence, sample_rate=sr))
    full_drum_audio = librosa.util.normalize(midi_synth.fluidsynth(combined_drum_sequence, sample_rate=sr))

    tap_and_onsets = mix_tracks(full_tap_audio, y[int(initial_start_time * sr):] / 2, stereo=True)
    drums_and_original = mix_tracks(full_drum_audio, y[int(initial_start_time * sr):] / 2, stereo=True)

    return full_drum_audio, full_tap_audio, tap_and_onsets, drums_and_original, combined_drum_sequence