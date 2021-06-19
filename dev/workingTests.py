import glob

print(len(glob.glob('../../data/train_set/*.dat')))
python scripts/train_mpii.py --arch=hg8 --image-path=/path/to/mpii/images --checkpoint=checkpoint/hg8 --epochs=220 --train-batch=6 --test-batch=6 --lr=5e-4 --schedule 150 175 200