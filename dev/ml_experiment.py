import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision import transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pims
import pathlib
import torch.optim as optim
from torch.autograd import Variable
import skimage as skm
import glob
import datetime
import yaml
import importlib
import random 
import hashlib
import time
import shutil
import csv
import subprocess
import os
import pickle
import dill

def pad(mask, val):
  for i in np.where(mask == val)[0]:
    for j in np.where(mask[i] == val)[0]:
      if i > 0 and j > 0 and i < len(mask) - 1 and j < len(mask[0]) - 1:
        mask[i-1][j-1] = val
        mask[i-1][j] = val
        mask[i-1][j+1] = val

        mask[i][j-1] = val
        mask[i][j] = val
        mask[i][j+1] = val

        mask[i+1][j-1] = val
        mask[i+1][j] = val
        mask[i+1][j+1] = val
  return mask


class DefectDataset(Dataset):
    """custom dataset for star locator"""

    def __init__(self, label_dir, root_dir, nn_type):
        """
        Args:
            label_dir (pathlib): folder containing labels
            root_dir (pathlib): root directory with all the images
            nn_type: string denoting what type of nn we are using (labels for unet and yolo are different)
        """
        self.root_dir = pathlib.Path(root_dir)
        self.label_dir = pathlib.Path(label_dir)
        self.labels = self.process_labels(nn_type)
        self.img_names = self.process_imgs()
        self.frames = pims.ImageSequence(str(self.root_dir / '*.tiff'))
        self.shape = self.frames[1].shape
        #ar = np.array(self.frames).reshape(-1,4)
        #self.meand = ar.mean(axis=0)
        #self.stdd = ar.std(axis=0)
        ar = None
        self.n = self.shape[0]
        #self.to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(tuple(self.meand[0:3]),tuple(self.stdd[0:3]))])
        self.to_tensor = transforms.ToTensor()

    def process_labels(self, nn_type):
        '''
        we need a function that will read in all the annotations and create a master dataframe [img_name][total_defect]
        nn_type: string, type of label for processing
        '''
        suffix = '*'+nn_type+'.dat'
        files = glob.glob(str(self.label_dir / suffix))
        df = pd.DataFrame([[f,np.genfromtxt(f)] for f in files], columns = ['names', 'numbers'])
        df['time'] = df['names'].str.extract('label_(\d*-\d*-\d*_\d*-\d*-\d)')
        df['frame'] = df['names'].str.extract('.*t_(\d*)').astype('int')
        df = df.sort_values(['time','frame']).reset_index(drop='True')
        #df.set_index('frame', inplace=True)
        return(df)

    def process_imgs(self):
        '''
        we need a function that will read in all the annotations and create a master dataframe [img_name] with the same order as the labels, so it can be indexed
        '''
        files = glob.glob(str(self.label_dir / '*.tiff'))
        df = pd.DataFrame(files, columns = ['names'])
        df['time'] = df['names'].str.extract('(\d*-\d*-\d*_\d*-\d*-\d)')
        df['frame'] = df['names'].str.extract('.*t_(\d*)\.tiff').astype('int')
        df = df.sort_values(['time','frame']).reset_index(drop='True')
        #df.set_index('frame', inplace=True)
        return(df)


    def __len__(self):
        return len(self.frames)
 

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx.tolist()
        img_name = self.img_names.iloc[idx]
        label_name = self.labels.iloc[idx]

        assert img_name['time'] == label_name['time']
        assert img_name['frame'] == label_name['frame']

        img = plt.imread(img_name['names'])
        label = label_name['numbers']

        img = skm.color.rgb2gray(img)
        img = skm.img_as_float(img)
        img = self.to_tensor(img)
        img = img.float()
        sample = [img, torch.from_numpy(label).float().unsqueeze(0)]
        return sample

def plt_mask(img, mask):
    plt.imshow(img.permute(1,2,0), cmap = 'gray')
    plt.imshow(crop_img(mask, img), alpha = .3)

class DataBlob():
    '''
    Create a tuple [train_dataset, test_dataset], so that we only have to initialize and load the once. Otherwise, each experiment would need to load its own data, which is a waste of time.
    '''
    def __init__(self, training_path, test_path, nn_type):
        self.trainSet = DefectDataset(training_path, training_path, nn_type)
        self.testSet = DefectDataset(test_path, test_path, nn_type)


def crop_img(tensor, target_tensor):

    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]

    delta = tensor_size - target_size
    delta = delta // 2
    tensor = tensor

    return tensor[:, :, delta:tensor_size - delta, delta:tensor_size - delta]


class Experiment():
    '''
    Will initialize a new experiment, with user defined net, training, testing, error function
    '''

    def __init__(self, unet_file, param_file, datablob, cost_func = nn.BCEWithLogitsLoss(), prototype=True):
        '''
        Args:
            unet_file: file containing definition of unet to use
            param_file: file containing experiment parameters
            datablob: data to run the experiment on
            cost_function: func to use as an error/cost function
            prototype: boolean to switch from testing to prod
        '''
        
        try:
            self.unet = importlib.import_module( str(pathlib.Path(unet_file).name.rstrip('.py')), package=None)
        except Exception as E:
            print(str(E)+': unet file doesnt exist!')

        with open(param_file, 'r') as config:
            self.cfg = yaml.safe_load(config)

        self.trainSet = datablob.trainSet
        self.testSet = datablob.testSet

        self.loss_fn = cost_func

        #set devic
        self.device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        print(f"Training on device {self.device}.")

        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight = torch.FloatTensor([10]).to(self.device))
        #load model and experiment parameters
        self.n_epochs = self.cfg['training']['n_epochs']
        self.optimizer = self.cfg['training']['optimizer']
        self.learning_rate = self.cfg['training']['learning_rate']
        self.model = self.unet.UNet().to(self.device)

        self.state = [] #the weights of the current unet. Will update every 10 epoch.
        self.log = []#log of current training (training_loss vs time)
        self.results = [] #summary of final training results. Accuracy, training loss.

        self.current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) 
        self.id = self.gen_hash(unet_file, param_file)
        self.hash_path = pathlib.Path('./experiments/'+self.id)
        self.hash_path.mkdir(parents = True, exist_ok = True)
        shutil.copy(unet_file, self.hash_path / pathlib.Path(unet_file).name)
        shutil.copy(param_file, str(self.hash_path / pathlib.Path(param_file).name))
        self.results_path = self.hash_path  / pathlib.Path(self.current_time)
        self.results_path.mkdir(parents = True, exist_ok = True)

        self.summary_file = self.results_path / pathlib.Path('summary.md')
        self.create_summary()
        self.state_file = self.results_path / pathlib.Path('state.md')
        self.loss_graph_name = self.results_path / pathlib.Path('loss.png')

        optim_dict = {'SGD' : optim.SGD(self.model.parameters(), lr = self.learning_rate, weight_decay = 0.0005, momentum = .9)}
        self.optimizer = optim_dict[self.cfg['training']['optimizer']]
        self.lr_gamma = self.cfg['training']['learning_update_gamma']

        scheduler_dict = {'expo': lambda epoch: self.lr_gamma ** epoch,
                          None: lambda epoch: 1}
        self.lr_schedule = scheduler_dict[self.cfg['training']['learning_schedule']]
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self.lr_schedule)

        self.batch_size = self.cfg['training']['batch_size']
        self.train_loader = torch.utils.data.DataLoader(self.trainSet, batch_size = self.batch_size, shuffle = True)
        self.test_loader = torch.utils.data.DataLoader(self.testSet, batch_size = self.batch_size, shuffle = False)
        self.total_epochs = 0 #keep a log of the total epochs trained

    def gen_hash(self, unet_file, param_file):
        val_hash = hashlib.md5()
        with open(unet_file, 'rb') as filename:
            model = filename.read()
            val_hash.update(model)
        
        with open(param_file, 'rb') as filename:
            params = filename.read()
            val_hash.update(params)

        return val_hash.hexdigest()

    def create_summary(self):
        #editor = os.environ.get('editor', 'vim')

        initial_message = "write a summary of this experiment:\n"
        summary = input('Please write a quick summary of this experiment:\n')

        with open(self.summary_file, 'w') as sum_file:
            sum_file.write(initial_message)
            sum_file.write(summary)
            #subprocess.call([editor, self.summary_file])
        return None

    def write_state(self):
        with open(self.state_file, 'wb') as state_file:
            dill.dump(self, state_file)

    def save_loss_graph(self):
        dat = pd.DataFrame(self.log, columns = self.header)
        self.loss_ax.clear()
        self.loss_ax.plot(dat['Epoch'], dat['Training loss'], label = 'train. loss')
        self.loss_ax.plot(dat['Epoch'], dat['Validation loss '], label = 'val. loss')
        self.loss_ax.legend(loc='best')
        self.loss_fig.savefig(self.loss_graph_name)
        



    def px_accuracy(self, output, label, thresh = .7):
        '''calculate the pixel accuracy of our predicted mask
        args:
            output: NxM tensor, output of model run through sigmoid
            label: NxM tensor, ground truth mask
            thresh: float, decision boundary to assign class output for model result
        '''
        assert(output.shape == label.shape)
        class_mask = output > thresh
        class_mask.int() #may not work with pytorch, works with np
        accur = (class_mask == label).float().mean()
        return accur
    
    def load_state(self, state_dict):
        '''
        load a previously trained model
        
        args:
            state_dict: path/state_dict
        '''
        if state_dict.__module__ == 'pathlib':
            try: 
                self.model.load_state_dict(torch.load(str(state_dict))) 
            except Exception as E:
                print(E)
                print("Check that the loaded model has the same parameters as the architecture")
        else:
            try: 
                self.model.load_state_dict(state_dict)
            except Exception as E:
                print(E)
                print('Did you specify state_dict as either a path or a loaded state_dict for a model?')

    def save_model(self):
        #first delete all old weight files
        old_train_list = glob.glob(str(self.results_path / pathlib.Path('*.weight')))
        for f in old_train_list:
            f_p = pathlib.Path(f)
            f_p.unlink()
        torch.save(self.model.state_dict(), str(self.results_path / pathlib.Path('trained_{}.weight'.format(self.total_epochs))))


    def save_graph(self, fig, ax_out, ax_labl, outputs_val, labels_val, save_g = True):
        ax_out.cla()
        ax_labl.cla()
        ax_out.imshow(torch.sigmoid(outputs_val).permute(1,2,0).squeeze().to('cpu').detach().numpy())
        ax_labl.imshow( crop_img(labels_val.unsqueeze(0), outputs_val.unsqueeze(0)).squeeze(0).permute(1,2,0).squeeze().to('cpu').detach().numpy())
        if save_g == True:
            fig.savefig(str(self.results_path / pathlib.Path('training_epoch_{}.png'.format(self.total_epochs))))

    def training_loop(self):
        '''
        training the nueral net
        '''

        self.header = ['Time ', ' ID' , 'Epoch' , 'Training loss' , 'Validation loss ',' pixel accuracy',  'std  ',' teststd']
        with open(self.results_path / pathlib.Path('log.dat'), 'a') as log_file:
                    writer = csv.writer(log_file)
                    writer.writerow(self.header)


        fig, [ax_out, ax_labl] = plt.subplots(nrows = 1, ncols = 2)
        self.loss_fig, self.loss_ax = plt.subplots()
        plt.ion()
        for epoch in range(1, self.n_epochs + 1):  # <2>
            self.total_epochs += 1
            loss_train = 0.0
            for imgs, labels in self.train_loader:  # <3>
                imgs = imgs.to(device=self.device) 
                labels = labels.to(device=self.device)
                outputs = self.model(imgs)  # <4>
                
                
                loss = self.loss_fn(outputs, crop_img(labels, outputs))  # <5>
                self.optimizer.zero_grad()  # <6>
                
                loss.backward()  # <7>
                
                self.optimizer.step()  # <8>

                loss_train += loss.detach().item()  # <9>


            if epoch == 1 or epoch % 10 == 0:
                with torch.no_grad():
                    loss_val = 0
                    for imgs_val, labels_val in self.test_loader:
                        imgs_val = imgs_val.to(device=self.device)
                        labels_val = labels_val.to(device=self.device)
                        outputs_val = self.model(imgs_val)
                        lossV = self.loss_fn(outputs_val, crop_img(labels_val, outputs_val))
                        loss_val += lossV.item()
                    self.state = self.model.state_dict()
                

                #output results and log
                out_temp =(datetime.datetime.now(),
                        self.id,
                        epoch,
                        loss_train/len(self.train_loader),
                        loss_val/len(self.test_loader),
                        self.px_accuracy(torch.sigmoid(outputs_val.detach()),
                        crop_img(labels_val, outputs_val.detach())),
                        outputs.std(),
                        labels.float().std())

                out_temp = [k.detach().item() if ( type(k) == torch.Tensor) else k for k in out_temp]
                self.log.append(out_temp)
                out_string = [h+': {}, '.format(o) for h,o in zip(self.header, out_temp)]
                print(*out_string)
                #save state and print test graph comparing output to ground truth
                self.save_model()
                self.save_loss_graph()
                with torch.no_grad():
                    plot_test_out = self.model(next(iter(self.test_loader))[0][0].unsqueeze(0).to(self.device))[0]
                    plot_test_label = next(iter(self.test_loader))[1][0]
                self.save_graph(fig, ax_out, ax_labl, plot_test_out, plot_test_label)

                with open(self.results_path / pathlib.Path('log.dat'), 'a') as log_file:
                    writer = csv.writer(log_file)
                    writer.writerow(out_temp)

                self.scheduler.step()


