import pandas as pd
import numpy as np
from pydub import AudioSegment
import soundfile
from train_label import label_extraction
import pickle


def segment_audio(filename, start, end):
    audio = AudioSegment.from_wav(filename)
    #.resample(sample_rate_Hz=16000, sample_width=2, channels=1)
    start = start * 1000
    end = end * 1000
    segment = audio[start:end]
    return segment


def remove_experimenter(filename, id_number):
    data = pd.read_csv(filename, header=None)
    data = data[0]
    # CHANGE TO YOUR AUDIO LOCATION

    aud_path = "data/" + str(all_ids[j]) + "_P/" + str(all_ids[j]) + "_AUDIO.wav"
    counter = 0
    for i in data:
        i = i.split()
        if "Participant" in i:
            start_time = i[0]
            end_time = i[1]
            try:
                segment = segment_audio(aud_path, float(start_time), float(end_time))
                # WHERE TO SAVE THE AUDIO FILE, sr is 16k right now
                soundfile.write("data/test_audio_by_uttr/spk_" + str(id_number) + "_uttr" + str(counter) + ".wav",
                                segment.get_array_of_samples(), 16000, subtype="PCM_16")
                counter += 1
            except ValueError:
                print("failed! skipping segment")
                print(i)
    return counter


# To save num utterances by id
all_counts = []

label_extraction()

# read the data from pickle file
with open('feature/test_label.pickle', 'rb') as handle:
    train_labels = pickle.load(handle)

# LIST OF PARTICIPANT IDS THAT YOU WANT TO SEGMENT
all_ids = list(train_labels.keys())

# all_ids = [480]

for j in range(0, len(all_ids)):
    # The audio transcript
    file = "data/" + str(all_ids[j]) + "_P/" + str(all_ids[j]) + "_TRANSCRIPT.csv"
    all_counts.append(remove_experimenter(file, all_ids[j]))

all_counts = np.array(all_counts)

np.save("num_utterances", all_counts)