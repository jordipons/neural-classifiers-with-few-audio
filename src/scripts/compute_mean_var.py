import shared, config_file
import json
import pickle
import numpy as np

AUDIO_REPRESENTATION_FOLDER = 'audio_representation/us8k__time-freq/'
INDEX_FILE = 'index.tsv'
PERCENTAGE_INDEX_FILE = 1 # 1 or 0.10

N_FRAMES = 128
PAD_SHORT = 'repeat-pad'
PRE_PROCESSING = 'logC'
AUDIO_REP_TYPE = 'time-freq'

def compute_mean_std(index_file,percentage_index_file):

    # get a percentage of the index0
    fgt = open(config_file.DATA_FOLDER + config['audio_representation_folder'] + index_file)
    num_lines = sum(1 for line in open(config_file.DATA_FOLDER + config['audio_representation_folder'] + index_file))

    tmp = np.array([])
    count = 0
    for line in fgt.readlines():

        # load audio representation
        id, audio_repr_path, audio_path = line.strip().split("\t")
        with open(config_file.DATA_FOLDER + audio_repr_path, 'rb') as f:
            audio_rep = pickle.load(f)

        # pre-process audio
        print(np.max(audio_rep))
        audio_rep = shared.pre_processing(audio_rep, N_FRAMES, PAD_SHORT, PRE_PROCESSING, AUDIO_REP_TYPE, normalize_mean=None, normalize_std=None)
        print(np.max(audio_rep))

        # append obs
        if count == 0:
            tmp = audio_rep
        else:
            tmp = np.concatenate((tmp,audio_rep), axis=0)
        print(tmp.shape)

        # do not store more data in memory
        # (to controll memory consumption)
        print(str(count) + '/' + str(num_lines))
        count = count + 1
        if count > num_lines*percentage_index_file:
            break

    print('Formatting data for computing mean - std!')
    data_sample = tmp.flatten()
    
    print('Computing mean:')
    mean = np.mean(data_sample)
    print(mean)
    
    print('Computing std:')
    std = np.std(data_sample)
    print(std)
    
    return mean, std


if __name__ == '__main__':

    # load config parameters used in 'audio_representation.py',
    config_json = config_file.DATA_FOLDER + AUDIO_REPRESENTATION_FOLDER + 'config.json'
    with open(config_json, "r") as f:
        params = json.load(f)
    config = params

    # compute mean var
    config['normalize_mean'], config['normalize_std'] = compute_mean_std(INDEX_FILE, PERCENTAGE_INDEX_FILE)

    # save config with mean and var
    json.dump(config, open(config_file.DATA_FOLDER + AUDIO_REPRESENTATION_FOLDER + "config.json", "w"))

