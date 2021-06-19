import torch
import ml_experiment

dat = ml_experiment.DataBlob('../../data/train_set/', '../../data/test_set/', 'gauss3')

e1 = ml_experiment.Experiment('unet.py', 'experiment.yaml', dat)

try:
    e1.training_loop()
except KeyboardInterrupt:
    print('ending train session.')
    e1.write_state()
