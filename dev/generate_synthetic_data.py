import defectData
import data_utils


# Generate n images using the masks defined in the maskTemplate.yaml file
#This will generate a random number of defects, then thermally evolve the sequence
#to give brownian flucuations. Then the augmentation masks will be applied.
res = 64
n = 2000
defectData.thermal_noise_sequence(n,res)

#Split the total data into a training and test split. Keep these physically seperate
#so that we can run fair comparisons across all of our models.
data_utils.train_test_split('../../data', '../../data')


#now, take the ground truth files, and create a mask on top of them (either a block mask (unet) or a gaussian kernal mask gauss)

#data_utils.create_gauss_mask('../train_set{}_sparse/'.format(res), n = 3)
#data_utils.create_gauss_mask('../test_set{}_sparse/'.format(res), n = 3)
data_utils.create_gauss_mask('../train_set{}_sparse/'.format(res), sig = 3)
data_utils.create_gauss_mask('../test_set{}_sparse/'.format(res), sig = 3)
#data_utils.create_unet_mask('../proto_test_set/'.format(res), n = 3)

#data_utils.proto_train_test_split('../../data/train_set', '../../data')