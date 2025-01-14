import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import pandas as pd
import os
import pickle
from scipy import stats


# label_csv = pd.read_csv(root_path + 'dev_split_Depression_AVEC2017.csv')
with open('../feature/test_label.pickle', 'rb') as handle:
    train_labels = pickle.load(handle)
user_list = list(train_labels.keys())

frame_size = int(16000*0.25)
overlap = 0.5

feature_dict = {}
for user in user_list:
    print(user)
    feature_dict[user] = []
    y, sr = librosa.load('../data/test_audio_by_uttr/' + 'spk_'+ str(user) + '_uttr0.wav', sr=16000)
    idx = 1
    filepath = '../data/test_audio_by_uttr/' + 'spk_'+ str(user) + '_uttr' + str(idx) + '.wav'
    while os.path.isfile(filepath):
        y, sr = librosa.load(filepath, sr=16000)
        start = 0
        # print(y.shape[0])
        while start + frame_size <= y.shape[0]:
            tmp = y[start:start + frame_size]
            start = start + int(frame_size*overlap)
            melspec = librosa.feature.melspectrogram(y=tmp, sr=sr, n_fft=512, hop_length=128, n_mels=128, center=False)
            logmelspec = librosa.power_to_db(melspec)
            feature_dict[user].append(logmelspec)

            # print(logmelspec.shape)
            # exit()
        # y = np.hstack((y, tmp))
        idx = idx + 1
        filepath = '../data/test_audio_by_uttr/' + 'spk_' + str(user) + '_uttr' + str(idx) + '.wav'


with open('../feature/test_spec.pickle', 'wb') as handle:
    pickle.dump(feature_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
