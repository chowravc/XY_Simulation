import glob
import numpy as np
import os
import pathlib
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import random
import skimage
import cv2
import skimage.color
from skimage.transform import AffineTransform, warp
import shutil
import re


'''
utilities for moving and normalizing training data. 
'''

class augMask():
    '''
    augMask: dynamically creates an augmentation mask, so when the raw spin grid is written to file, it looks more like the experimental data.

    It will read in a yaml file, which will tell it which submasks to use (gaussian noise, standardization, skewing, smart noise, grid noise, circle noise, etc) and what parameters to feed to those submasks. Every time the mask is called, it will give different results, as each submask will have a different random parameter.

    args:
        config_file: string, pointing to the yaml file which contains the schematic for constructing the mask out of the submasks

    config_file: yaml file with the following layout:

    submasks:
        submask1:
            use: True
            params:
                p1:
                p2:

        submask2:
        ...
    '''
    
    def __init__(self,config_file):
        
        self.config_file = config_file

        with open(self.config_file, 'r') as config:
            self.cfg = yaml.safe_load(config)
        self.dims = self.cfg['image_dimensions']
        self.submasks = self.cfg['submasks']
        self.mask_list = [e for e in self.submasks if self.submasks[e]['use'] == True] #read in config file, create list of masks which user wants to run
        print('submasks: '+str(self.mask_list))
        self.mask = self.sub_mask_assemble()
#        self.test_image = plt.imread('../data/2020-09-11_09-44-36-0.tiff')
        

    def sub_mask_assemble(self):

        def out(image):
            return skimage.color.rgb2gray(skimage.img_as_float(image))


        if 'gauss' in self.mask_list:
            out = self.gauss(out)
        if 'smart' in self.mask_list:
            out = self.smart(out)
        if 'skew' in self.mask_list:
            out = self.skew(out)
        if 'circle' in self.mask_list:
            out = self.circle(out)
        if 'grid' in self.mask_list:
            out = self.grid(out)
        if 'scans' in self.mask_list:
            out = self.scans(out)
        if 'standardize' in self.mask_list:
            out = self.standardize(out)
        if 'gradient' in self.mask_list:
            out = self.gradient(out)
        if 'speckle' in self.mask_list:
            out = self.speckle(out)

        return out

    def gauss(self, func):

        params = self.submasks['gauss']['params']
        sigma_min = params['min']
        sigma_max = params['max']

        #print('gaussianing')
        def gauss_wrap(*args, **kwargs):
            return gaussian_filter(func(*args, **kwargs), sigma=random.uniform(sigma_min, sigma_max))
        return gauss_wrap
    

    def standardize(self, func):
                     
        #print('standarizing')
        def standard_wrap(*args, **kwargs): 
            image = func(*args, **kwargs)
            mean = image.mean()
            std = image.std()
            l = (func(*args, **kwargs)-mean)/std
            return (func(*args, **kwargs)-mean)/std
                     
        return standard_wrap

    def skew(self, func, shift):
        
        def skew_wrapper(*args, **kwargs):
            transform = AffineTransform(translation=shift)
            return warp(func(*args, **kwargs), transform, mode = 'wrap', preserve_range = True).astype(float)


        return skew_wrapper

    def skewImage(self, image, shift):

        def simp(arg):
            return arg
        out = self.skew(simp, shift)
        return out(image)


        return None
    
    def smart(self, func):
        #print('smarting')
        smartNoiseMax = self.submasks['smart']['params']['strengthMax']
        smartNoiseMin = self.submasks['smart']['params']['strengthMin']
        pathToNoise = self.submasks['smart']['params']['pathToNoise']
        noiseImages = [plt.imread(f) for f in glob.glob(os.getcwd()+pathToNoise+'*.jpg')]

        def smart_wrapper(*args, **kwargs):
            noiseImage = noiseImages[0]#noiseImages[random.randint(0,len(noiseImages))-1]
            noiseImage = skimage.color.rgb2gray(skimage.img_as_float(noiseImage))
            standardMask = self.standardize(lambda x: x) #need to add weird lambda, as standardize thinks its wrapping other functions. Give it a function that returns its argument to make it happy.
            noiseImage = standardMask(noiseImage)
            nDims = noiseImage.shape
            gDims = self.dims
            xSkew = random.randint(0,nDims[0])
            ySkew = random.randint(0,nDims[1])
            noiseImage = self.skewImage(noiseImage,(xSkew,ySkew))
            if nDims[0]>=gDims[0] and nDims[1]>=gDims[1]:
                noiseImage = noiseImage[0:gDims[0],0:gDims[1]]
            else:
                noiseImage = cv2.resize(noiseImage, dsize=(gDims[0], gDims[1]), interpolation=cv2.INTER_CUBIC)
            noiseStrength = random.uniform(smartNoiseMin,smartNoiseMax)
            return func(*args, **kwargs)+noiseImage*noiseStrength
        return smart_wrapper

 

    def gradient(self, func):
        #print('gradient')
        lower_bound = self.submasks['gradient']['params']['lower_bound']
        def grad_wrap(*args, **kwargs):
            theta = random.uniform(0, 2*np.pi)
            strength = random.uniform(lower_bound, 1)
            xx, yy = np.mgrid[0:1:np.complex(0,self.dims[0]), 0:1:np.complex(0,self.dims[1])]
            xx = np.cos(theta)*xx+np.sin(theta)*yy
            yy = np.cos(theta)*yy-np.sin(theta)*xx
            mask = xx*yy

            mask = (mask-mask.min())
            mask = mask/mask.max()

            mask = mask*(1-strength)
            mask = mask+strength

            return func(*args, **kwargs)*mask

        return grad_wrap

         


    def scans(self, func):
        #print('scanning')
        strength = self.submasks['scans']['params']['strength']

        def scan_wrap(*args, **kwargs):
            mask = np.zeros(self.dims)
            for i in range(0,self.dims[0]):
                mask[i,:,] = random.uniform(-strength,strength)
            for i in range(0,self.dims[1]):
                mask[:,i] = random.uniform(-strength,strength)
            return func(*args, **kwargs)+mask
        
        return scan_wrap

    def speckle(self, func):
        #print('speckling')

        max_speckles = self.submasks['speckle']['params']['maxSpeckles']
        max_area = self.submasks['speckle']['params']['maxArea']
        speckle_strength = self.submasks['speckle']['params']['speckleStrength']

        def speckle_wrap(*args, **kwargs):
            area = random.randint(1, max_area)
            num_speckles = random.randint(1,max_speckles)
            mask = np.zeros(self.dims)
            for speck in range(num_speckles):
                initial_x = random.randint(0,self.dims[0]-1)
                initial_y = random.randint(0,self.dims[1]-1)
                
                mask[initial_x, initial_y] = -1

                px_x = initial_x
                px_y = initial_y

                for pixel in range(area):
                    
                    #d_x = random.sample([-1,1], 1)[0] #choose random direction for x
                    #d_y = random.sample([-1,1], 1)[0] #choose random direction for y     
                    d_x = pixel*np.mod(pixel, 2)
                    d_y = pixel*(np.mod(pixel, 2)+1)

                    px_x = np.mod(px_x + d_x, self.dims[0]) #update new pixel, set mask equal to zero
                    px_y = np.mod(px_y + d_y, self.dims[0])
                    mask[px_x, px_y] = -1
                    mask[px_x, px_y] = -1
            #print(speckle_strength*mask.min())
                

            return speckle_strength*mask +func(*args, **kwargs)
        return speckle_wrap


    def circle(self, func):
        #print('circling')
        num_circles = self.submasks['circle']['params']['numCircles']
        radiusRange = self.submasks['circle']['params']['radiusRange']
        brightnessRange = self.submasks['circle']['params']['brightnessRange']

        def circle_mask(radius, xCoord, yCoord):
            dims = self.dims
            mask = np.zeros(dims)
            xx,yy = np.mgrid[:dims[0],:dims[1]]

            circle = radius>((xx - xCoord)**2+(yy-yCoord)**2)**(1/2)

            mask[circle] = random.uniform(*brightnessRange)
            return mask

        def circle_wrap(*args, **kwargs):
            mask = np.zeros(self.dims)
            for i in range(num_circles):
                radius = random.randint(*radiusRange)
                xCoord = random.randint(0,self.dims[0])
                yCoord = random.randint(0,self.dims[1])
                m1 = circle_mask(radius, xCoord, yCoord)
                mask = mask+circle_mask(radius, xCoord, yCoord)
            return mask+func(*args, **kwargs)

        return circle_wrap

    def grid(self, func):
        #print('gridding')
        bShift = self.submasks['grid']['params']['gridRange']
        dims = self.dims
        def grid_wrap(*args, **kwargs):
            x = random.randint(0,dims[0])
            y = random.randint(0,dims[1])
            mask = np.zeros(dims)
            mask[:x,:y] = mask[:x,:y] +random.uniform(-bShift,bShift)
            mask[x:,:y] = mask[x:,:y] +random.uniform(-bShift,bShift)
            mask[:x,y:] = mask[:x,y:] +random.uniform(-bShift,bShift)
            mask[:x,:y] = mask[:x,:y] +random.uniform(-bShift,bShift)
 
            return func(*args, **kwargs)+mask
 
        return grid_wrap



def train_test_split(img_path, label_path, ratio = .8):
    '''
    Seperate a set of images and labels into a test-train split, labeled by directories: test and train
    args:
        img_path: pathlib path to directory containing all the images (assuming .tiff files)
        label_path: pathlib to directory containing all the labels (assuming .dat files)
        ratio: ratio of test/train split

    The default behaviour will be to create two new directories where the images used to be, and then randomly
    and place the images and labels in one or the other
    '''

    #args
    #img_path = '../data'
    #label_path = '../data'

    if type(img_path) == str:
        img_path = pathlib.Path(img_path)

    if type(label_path) == str:
        label_path = pathlib.Path(label_path)

    img_list = glob.glob(str(img_path)+'/*.tiff')
    label_list = glob.glob(str(label_path)+'/*.dat')
    dim = plt.imread(img_list[0]).shape[0]

    test_dir = img_path / 'test_set{}_sparse/'.format(dim)
    train_dir = img_path / 'train_set{}_sparse/'.format(dim)

    if test_dir.is_dir() == False:
        os.mkdir(str(test_dir))
    else:
        print('directory {} already exists!'.format(str(test_dir)))

    if train_dir.is_dir() == False:
        os.mkdir(str(train_dir))
    else:
        print('directory {} already exists!'.format(str(train_dir)))


    #glob won't garuntee that the index of the label and img will match. Change into series for each, extract the date and the frame number, sort on the date, then frame. This will align the indexes of each series (frame and date will be the same for each img and label). Then we can do a simple index randomization, and move each one where it should go.

    img_frame = pd.DataFrame(img_list, columns = ['name'])
    label_frame = pd.DataFrame(label_list, columns = ['name'])

    img_frame['time'] = pd.to_datetime(img_frame['name'].str.extract('(\d*-\d*-\d*_\d*-\d*-\d*)', expand = False), format = '%Y-%m-%d_%H-%M-%S')
    label_frame['time'] = pd.to_datetime(label_frame['name'].str.extract('(\d*-\d*-\d*_\d*-\d*-\d*)', expand = False), format = '%Y-%m-%d_%H-%M-%S')

    img_frame['frame'] = img_frame['name'].str.extract('t_(\d*)', expand = False).astype(int)
    label_frame['frame'] = label_frame['name'].str.extract('t_(\d*).dat', expand = False).astype(int)
    label_frame['class'] = 'label'
    img_frame['class'] = 'img'

    img_frame = img_frame.sort_values(['time', 'frame'])
    label_frame = label_frame.sort_values(['time', 'frame'])

    #if there is a misnamed label, then this will throw everything off, but really hard to catch (labels will be off by one)

    #combine the two dataframes into one, convert to multi-index, then select on the iloc

    df = pd.concat([img_frame, label_frame])

    move_group = df.groupby(['time','frame'])

    num_imgs = move_group.ngroups
    idx8 = int(num_imgs*.8)
    idx2 = num_imgs - idx8
    index = np.r_[np.zeros(idx8),np.ones(idx2)] #assuming .8/.2 split (1/.8 to take advantage of divide cast)
    np.random.shuffle(index)
    i = 0
    for time_frame, time_frame_gp in move_group:
        assert len(time_frame_gp) == 2

        if index[i] == 0:
            move_name = train_dir
        else:
            move_name = test_dir
        in_name_img = time_frame_gp['name'][time_frame_gp['class'] == 'img']
        in_name_label = time_frame_gp['name'][time_frame_gp['class'] == 'label']

        in_name_img = pathlib.Path(in_name_img.iloc[0])
        in_name_label = pathlib.Path(in_name_label.iloc[0])

        dest_img = move_name / in_name_img.name
        dest_label = move_name / in_name_label.name

        if not dest_img.exists():
            in_name_img.replace(dest_img)
        if not dest_label.exists():
            in_name_label.replace(dest_label)

        i += 1



#   return df

def create_gauss_mask(src_path, sig = 3): 
    '''
    Take in labels where each defect pixel position is labelled by a 1, and create an nxn mask, for use in the unet.
    args:
        src_path: where the labels live
        n: how big of a mask to create (default 3x3)
    '''
    
    def pad(mask, val=1):
        #initialize gaussian mask
        m_ar = np.zeros(mask.shape)

        for i,j in zip( *np.where(mask == val) ):
            row_ar, col_ar = np.indices(mask.shape)
            row_ar = row_ar - i
            col_ar = col_ar -j
            m_ar = m_ar + np.exp(- (row_ar**2/2./sig**2) - (col_ar**2/2./sig**2))

        return m_ar

    dat_files = glob.glob(src_path+'*.dat')
    raw_files = [ l for l in dat_files if re.search('t_\d*\.dat', l)]
    for name in raw_files:
        print(name)
        ar = np.abs(np.genfromtxt(name))
        ar = pad(ar)
        np.savetxt(name.replace('.dat', '_gauss{}.dat'.format(sig)) , ar)





#    return df

def create_unet_mask(src_path, n = 3):
    '''
    Take in labels where each defect pixel position is labelled by a 1, and create an nxn mask, for use in the unet.
    args:
        src_path: where the labels live
        n: how big of a mask to create (default 3x3)
    '''
    
    def pad(mask, val=1):
        for i,j in zip( *np.where(mask == val) ):
           if i > 0 and j > 0  and i < len(mask) - 1 and j < len(mask[0])-1:
               spread = np.arange(n)
               spread = spread- spread[n//2]
               for k in spread:
                   for l in spread:
                       mask[i+k][j+l] = 1
        return mask

    dat_files = glob.glob(src_path+'*.dat')
    raw_files = [ l for l in dat_files if re.search('t_\d*\.dat', l)]
    for name in raw_files:
        print(name)
        ar = np.abs(np.genfromtxt(name))
        ar = pad(ar)
        np.savetxt(name.replace('.dat', '_unet{}.dat'.format(n)) , ar)



def proto_train_test_split(src_path, out_path, ratio = .8):
    '''
    Seperate a set of images and labels into a test-train split, labeled by directories: test and train
    args:
        src_path: pathlib path to directory containing all the images/labels
        out_path: pathlib path to directory where you want to save this
        ratio: ratio of test/train split

    The default behaviour will be to create two new directories where the images used to be, and then randomly
    and place the images and labels in one or the other
    '''

    #args
    #img_path = '../data'
    #label_path = '../data'
    np.random.seed(40)
    if type(src_path) == str:
        src_path = pathlib.Path(src_path)

    if type(out_path) == str:
        out_path = pathlib.Path(out_path)

    img_list = glob.glob(str(img_path)+'/*.tiff')
    label_list = glob.glob(str(label_path)+'/*.dat')
    dim = plt.imread(img_list[0]).shape[0]

    test_dir = img_path / 'proto_test_set{}/'.format(dim)
    train_dir = img_path / 'proto_train_set{}/'.format(dim)



    if test_dir.is_dir() == False:
        os.mkdir(str(test_dir))
    else:
        print('directory {} already exists!'.format(str(test_dir)))

    if train_dir.is_dir() == False:
        os.mkdir(str(train_dir))
    else:
        print('directory {} already exists!'.format(str(train_dir)))

    img_list = glob.glob(str(src_path)+'/*.tiff')
    label_list = glob.glob(str(src_path)+'/*.dat')

    #glob won't garuntee that the index of the label and img will match. Change into series for each, extract the date and the frame number, sort on the date, then frame. This will align the indexes of each series (frame and date will be the same for each img and label). Then we can do a simple index randomization, and move each one where it should go.

    img_frame = pd.DataFrame(img_list, columns = ['name'])
    label_frame = pd.DataFrame(label_list, columns = ['name'])

    img_frame['time'] = pd.to_datetime(img_frame['name'].str.extract('(\d*-\d*-\d*_\d*-\d*-\d*)', expand = False), format = '%Y-%m-%d_%H-%M-%S')
    label_frame['time'] = pd.to_datetime(label_frame['name'].str.extract('(\d*-\d*-\d*_\d*-\d*-\d*)', expand = False), format = '%Y-%m-%d_%H-%M-%S')

    img_frame['frame'] = img_frame['name'].str.extract('t_(\d*)', expand = False).astype(int)
    label_frame['frame'] = label_frame['name'].str.extract('t_(\d*)', expand = False).astype(int)
    label_frame['class'] = 'label'
    img_frame['class'] = 'img'

    img_frame = img_frame.sort_values(['time', 'frame'])
    label_frame = label_frame.sort_values(['time', 'frame'])

    #if there is a misnamed label, then this will throw everything off, but really hard to catch (labels will be off by one)

    #combine the two dataframes into one, convert to multi-index, then select on the iloc

    df = pd.concat([img_frame, label_frame])

    move_group = df.groupby(['time','frame'])

    num_imgs = 200 #number of total images
    idx8 = int(num_imgs*.8)
    idx2 = num_imgs - idx8
    index = np.r_[np.zeros(idx8),np.ones(idx2)] #assuming .8/.2 split (1/.8 to take advantage of divide cast)
    np.random.shuffle(index)
    print(len(move_group))
    rand_indx = np.arange(len(move_group))

    np.random.shuffle(rand_indx)
    group_index = list(move_group.groups.keys())
    for i, k in enumerate(rand_indx[0:num_imgs]):
        time_frame_gp = move_group.get_group(group_index[k])

        if index[i] == 0:
            move_name = train_dir
        else:
            move_name = test_dir

        in_name_img_list = time_frame_gp['name'][time_frame_gp['class'] == 'img']
        in_name_label_list = time_frame_gp['name'][time_frame_gp['class'] == 'label']

        for in_name_img in in_name_img_list:

            in_name_img_path = pathlib.Path(in_name_img)
            dest_img = move_name / in_name_img_path.name
            if not dest_img.exists():
                shutil.copy(str(in_name_img_path), str(dest_img))
        
        for in_name_label in in_name_label_list:
            in_name_label_path = pathlib.Path(in_name_label)
            dest_label = move_name / in_name_label_path.name

            if not dest_label.exists():
                shutil.copy(in_name_label_path, dest_label)


