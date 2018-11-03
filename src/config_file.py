import numpy as np

N_SHOT = 100                                      # how many training examples do we allow per class? (n data points)
RUNS = 1                                          # how many times do we want to run the experiment? (m runs)
FOLD = 0                                          # in which fold? For ASC-TUT, the only fold is fold0.

DATASET = 'asc'                                   # 'us8k' for US8K or 'asc' for ASC-TUT
NUM_CLASSES = 15                                  # 10 if 'us8k' (US8K) and 15 if 'asc' (ASC-TUT)
DATA_FOLDER = '/home/jpons/Github/telefonica/data/'      # set your data folder

config_preprocess = {
    'us8k_spec': {
        'identifier': 'us8k',                     # name for easy identification
        'audio_folder': '/home/jpons/audio/UrbanSound8K/audio/', # end it with / -> this is an absolute path!
        'n_machines': 1,                          # parallelizing this process through 'n_machines'
        'machine_i': 0,                           # id number of the machine which is running this script (from 0 to n_machines-1)
        'num_processing_units': 1,                # number of parallel processes in every machine
        'type': 'time-freq',                      # kind of audio representation: 'time-freq' or 'waveform' or 'audioset'
        'spectrogram_type': 'mel',                # 'mel', 'cqt', 'stft' - parameters below should change according to this type
        'resample_sr': 44100,                     # sampling rate (original or the one to be resampled)
        'hop': 1024,                              # hop size of the STFT
        'n_fft': 1024,                            # number of freq bins of the STFT
        'n_mels': 128,                            # number of mel bands
        'convert_id': False,                      # converts the (path) name of a file to its ID name, correspondence defined in index_file
        'index_file': 'index/us8k/index_us8k_folds_all.tsv', # list of audio representations to be computed
    },
    'asc_spec': {
        'identifier': 'asc',	                  
        'audio_folder': '/home/jpons/audio/ASC-TUT/',             
        'n_machines': 1,  		          
        'machine_i': 0,  		          
        'num_processing_units': 1,  	          
        'type': 'time-freq',  		          
        'spectrogram_type': 'mel',  	          
        'resample_sr': 44100,  		          
        'hop': 1024,  	    		          
        'n_fft': 1024,  		          
        'n_mels': 128,  		          
        'convert_id': False,  	 	          
        'index_file': 'index/asc/index_asc_folds_all.tsv',
    },
    'us8k_audioset': {
        'identifier': 'us8k',	                  
        'audio_folder': '/home/jpons/audio/UrbanSound8K/audio/',
        'n_machines': 1,  		          
        'machine_i': 0,  		          
        'num_processing_units': 1,  	          
        'type': 'audioset',  		          
        'convert_id': False,  	 	          
        'index_file': 'index/us8k/index_us8k_folds_all.tsv', 
    },
    'asc_audioset': {
        'identifier': 'asc',
        'audio_folder': '/data/ASC/',
        'n_machines': 1,
        'machine_i': 0,
        'num_processing_units': 1,
        'type': 'audioset',
        'convert_id': False,
        'index_file': 'index/asc/index_asc_folds_all.tsv',
    }
}

config_sl = {
    'spec': {
        # which data?
        'audio_representation_folder': 'audio_representation/'+str(DATASET)+'__time-freq/',
        'gt_train': 'index/'+str(DATASET)+'/gt_'+str(DATASET)+'_fold'+str(FOLD)+'_train.tsv',
        #'gt_val': 'index/'+str(DATASET)+'/gt_'+str(DATASET)+'_fold'+str(FOLD)+'_val.tsv', # set this line to run sl_train_val.py

        # input setup?
        'n_frames': 128,                          # if '', compute n_frames from 'window'. Set an INT otherwise!
        'pre_processing': 'logEPS',               # 'logEPS' or None
        'pad_short': 'repeat-pad',                # 'zero-pad' or 'repeat-pad'
        'train_sampling': 'random',               # 'overlap_sampling' or 'random'. how to sample patches from the audio?
        'param_train_sampling': 1,                # if mode_sampling='overlap_sampling': param_sampling=hop_size
                                                  # if mode_sampling='random': param_sampling=number of samples
        # learning parameters?
        'model_number': 3,                        # number of the model as in models_sl.py
        'load_model': None,                       # set to 'None' or absolute path to the model
        'epochs': 200,                            # maximum number of epochs before stopping training
        'batch_size': 256,                        # batch size during training
        'learning_rate': 0.1,                     # learning rate
        'weight_decay': 0.001,                    # None or value for the regularization parameter (0.001)
        'optimizer': 'SGD_clip',                  # 'SGD_clip'
   
        # experiment settings?
        'n_shot': N_SHOT,
        'num_experiment_runs': RUNS,
        'num_classes_dataset': NUM_CLASSES,
        'val_batch_size': 256
    }
}


config_transfer = {
    'audioset': {
        # experiment settings?
        'n_shot': N_SHOT,
        'num_experiment_runs': RUNS,
        'num_classes_dataset': NUM_CLASSES,
        'name_run': '',

        # which data?
        'audio_representation_folder': 'audio_representation/'+str(DATASET)+'__audioset/',
        'gt_train': 'index/'+str(DATASET)+'/gt_'+str(DATASET)+'_fold'+str(FOLD)+'_train.tsv',
        #'gt_val': 'index/'+str(DATASET)+'/gt_'+str(DATASET)+'_fold'+str(FOLD)+'_val.tsv', # uncomment to run transfer_train_val.py

        # input setup?
        'train_sampling': 'random',
        'param_train_sampling': 1,

        # learning parameters?
        'epochs': 200,                            # maximum number of epochs before stopping training
        'batch_size': 128,                        # batch size during training
        'learning_rate': 0.1,                     # learning rate
        'optimizer': 'many_lr_audioset',          # 'many_lr_audioset' or 'SGD_clip'
   
        'val_batch_size': 256
    }
}


config_proto = {
    'spec': {
        # experiment settings?
        'n_shot': N_SHOT,
        'num_experiment_runs': RUNS,
        'num_classes_dataset': NUM_CLASSES,
        'name_run': '',

        # which data?
        'audio_representation_folder': 'audio_representation/'+str(DATASET)+'__time-freq/',
        'gt_train': 'index/'+str(DATASET)+'/gt_'+str(DATASET)+'_fold'+str(FOLD)+'_train.tsv',
        'gt_test': 'index/'+str(DATASET)+'/gt_'+str(DATASET)+'_fold'+str(FOLD)+'_test.tsv',

        # input setup?
        'n_frames': 128,                          # set as int
        'pre_processing': 'logEPS',               # 'logEPS' or None
        'pad_short': 'repeat-pad',                # 'zero-pad' or 'repeat-pad'
        'train_sampling': 'overlap_sampling',     # 'overlap_sampling' or 'random' - how to sample patches from the song
        'param_train_sampling': 42,               # if mode_sampling='overlap_sampling': param_sampling=hop_size
                                                  # if mode_sampling='random': param_sampling=number of samples
        # model parameters?
        'model_number': 2,                        # number of the model as in models_proto.py
        'load_model': None,			  # set to 'None' or absolute path to the model

        # when learning..
        'optimizer': 'SGD_clip',                  # 'SGD_clip'
        'learning_rate': 0.1,                     # learning rate
        'max_patches_per_class': 5,               # batch-size per class during training               
        'patches_per_prototype': 5,               # number of patches per class used to compute the prototype (support set)
        'max_epochs': np.inf,                     # maximum number of epochs before the model stops training
        'max_accuracy': 200,                      # number of times the model does not improve before stop training (stop criteria)

        'preprocess_batch_size': 128,             # max batch size that fits into memory
        'test_batch_size': 128,                   # max batch size that fits into memory

    }
}


config_transfer_proto = {
    'audioset': {
        # experiment settings?
        'n_shot': N_SHOT,
        'num_experiment_runs': RUNS,
        'num_classes_dataset': NUM_CLASSES,
        'name_run': '',

        # which data?
        'audio_representation_folder': 'audio_representation/'+str(DATASET)+'__audioset/',
        'gt_train': 'index/'+str(DATASET)+'/gt_'+str(DATASET)+'_fold'+str(FOLD)+'_train.tsv',
        'gt_test': 'index/'+str(DATASET)+'/gt_'+str(DATASET)+'_fold'+str(FOLD)+'_test.tsv',

        # input data setup?
        'train_sampling': 'overlap_sampling',
        'param_train_sampling': 1,

        # when learning..
        'optimizer': 'many_lr_audioset',          # 'many_lr_audioset' or 'SGD_clip'
        'learning_rate': 0.1,                     # learning rate
        'max_patches_per_class': 5,               # batch-size per class during training
        'patches_per_prototype': 5,               # number of patches per class used to compute the prototype (support set)
        'max_epochs': np.inf,                     # maximum number of epochs before the model stops training
        'max_accuracy': 200,                      # number of times the model does not improve before stop training (stop criteria)

        'preprocess_batch_size': 64,
        'test_batch_size': 64,

    }
}
