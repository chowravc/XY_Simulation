'''
generate training data for xy model predictions
'''

import numpy as np
import pathlib
import pandas as pd
import glob
import os
import yaml
import h5py
import time
import langanLib
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
import time
import data_utils

def decrossI(beta,image):
#beta is the angle between pol and anl (90 for completely crossed)
    temp= ( np.sin(image)*np.cos(image)*np.sin(beta)-np.sin(image)**2*np.cos(beta)-np.cos(beta))**3
    return temp/temp.max()


def thermal_noise_sequence(n_imgs, res):

    # Opening relevant config file
    cfgFilepath = './thermalNoiseParams.yaml'
    with open(cfgFilepath, 'r') as config:
        thermalNoiseParams = yaml.safe_load(config)

    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    number_decrosses = thermalNoiseParams['params']['image_utils']['number_decrosses']
    number_augments = thermalNoiseParams['params']['image_utils']['number_augments']
    simulation_iterations = thermalNoiseParams['params']['image_utils']['simulation_iterations']
    snapshots = thermalNoiseParams['params']['image_utils']['snapshots'] #doesn't really matter

    image_dimensions = [res,res]

    betaMax = thermalNoiseParams['params']['beta']['max']
    betaMin = thermalNoiseParams['params']['beta']['min'] # decross angle
    maxDefect = thermalNoiseParams['params']['defects']['max'] # image_dimensions[0] // 4
    minDefect = thermalNoiseParams['params']['defects']['min']
    mask = data_utils.augMask('./maskTemplate.yaml')
    
    for i in np.arange(0,n_imgs,number_decrosses*number_augments):
        print('{} %'.format(float(i/n_imgs)*100))
        n_defects = random.randint(minDefect,maxDefect)
        t = Texture(n_defects, simulation_iterations, snapshots, image_dimensions, thermalNoiseParams)
        for j in np.arange(0,number_decrosses):
            beta = random.uniform(betaMin, betaMax)
            for k in np.arange(0,number_augments):
                out_img = mask.mask(decrossI(beta, t.xy.snapshots['lattice'][-1]))
                out_defect = t.xy.snapshots['defects'][-1]

                plt.imsave('../../data/'+current_time+'-t_{}.tiff'.format(i*number_decrosses*number_augments+j*number_augments+k), out_img, cmap = 'gray')
                np.savetxt('../../data/label_'+current_time+'-t_{}.dat'.format(i*number_decrosses*number_augments+j*number_augments+k), out_defect)



class Texture():
    def __init__(self, defect_number, n_iterations, n_pictures, size, thermalNoiseParams):
        self.defect_number = defect_number
        self.n_iterations = n_iterations
        self.n_pictures = n_pictures
        self.size = size
        self.thermalNoiseParams = thermalNoiseParams

        self.texture = np.zeros(size)
        self.gen_defects()
        #self.texture = np.random.random(size)*2*np.pi
        self.initialState = self.texture[:]
        self.xy = self.evolve()

    def view(self):
            def decrossI(beta,image):
            #beta is the angle between pol and anl (90 for completely crossed)
                temp= ( np.sin(image)*np.cos(image)*np.sin(beta)-np.sin(image)**2*np.cos(beta)-np.cos(beta))**3
                return temp/temp.max()

            def schler(texture):
                return np.sin(2*texture)**2

            plt.imshow(schler(self.texture),cmap = 'gray')

    def gen_defects(self):
        ix,iy = np.indices(self.size)
        def d_gen(grid,x,y,k,off):
            grid = np.mod(grid+k*np.arctan2(ix-x, iy-y)+off,2*np.pi)
            return grid
        grid = self.texture

        cluster_range = self.thermalNoiseParams['params']['cluster_range']
        cluster = cluster_range*self.size[0]/self.defect_number*2 # Changes the average defect position, changes dynamically with no. of defect.
        for i in range(self.defect_number):
            dxp = random.randint(0,self.size[0]-1) # Position defect coordinate
            dyp = random.randint(0,self.size[0]-1)

            dxn = np.mod(dxp + random.sample([-1,1],1)[0]*np.random.poisson(lam = cluster), self.size[0]-1) # Negative defect coordinate
            dyn = np.mod(dyp + random.sample([-1,1],1)[0]*np.random.poisson(lam = cluster), self.size[0]-1)
            #dyn = random.randint(0,self.size[0]-1)
            
            grid = d_gen(grid, dxp,dyp,1, random.random()*2*np.pi)
            grid = d_gen(grid, dxn,dyn,-1, random.random()*2*np.pi)
        self.texture = grid
        return grid
    
    def evolve(self):
        self.initialState = np.asfortranarray(self.texture[:,:])

        beta = 1/self.thermalNoiseParams['params']['evolve']['temp'] # This is the temperature (1/temperature)
        stepSize = self.thermalNoiseParams['params']['evolve']['stepSize']
        something = self.thermalNoiseParams['params']['evolve']['something']

        xy = xyModelFort(self.size[0], 1, 0.0, beta, 0, self.n_pictures, self.n_iterations, 'random', stepSize, something) #Thermal run

        #Play with the 2variables. 0.02 is the step size, 10 is something
        xy.setGrid(self.initialState)
        xy.simRun(savegrid=False, savedefect=False, saveHam=False)
        self.texture = np.ascontiguousarray(xy.snapshots['lattice'][-1]) # Converts fortran array to numpy
        return xy


def schler(im):
    return np.sin(2*im)**2
class SimData():
    '''This defines the SimData object. Given a series of parameters, it will generate a series of images for training a neural net. The parameters should be stored in a config file called simData.yaml in the yaml format
    '''

    def __init__(self, config = '../data/simData.yaml'):

        self.config_file = config
        with open(self.config_file, 'r') as config:
            self.cfg = yaml.safe_load(config)

        self.n = int(self.cfg['generate']['n'])
        self.kappa = float(self.cfg['generate']['kappa'])
        self.mu = float(self.cfg['generate']['mu'])
        self.beta = float(self.cfg['generate']['beta'])
        self.alpha = float(self.cfg['generate']['alpha'])
        self.plots = int(self.cfg['generate']['plots'])
        self.it = int(self.cfg['generate']['it'])
        self.initScheme = self.cfg['generate']['initScheme']
        self.deltime = float(self.cfg['generate']['deltime'])
        self.frame_n = int(self.cfg['generate']['frame_n'])
        self.current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


    def gen(self):
        xy = xyModelFort(self.n, self.kappa, self.mu, self.beta, self.alpha, self.plots, self.it, self.initScheme, self.deltime, self.frame_n, current_time = self.current_time)
        xy.initGrid(xy.initScheme)
        xy.simRun(savegrid=True, savedefect=True, saveHam=False)
        return xy
    def gen_training(self, grid = None, dgrid = None):
        '''
        This will take the hdf5 files and 'unpack' them into tif images and labelled grids for training data
        '''
        if grid == None:
            grid = '../data/xy-'+self.current_time+'.hf'
        if dgrid == None:
            dgrid = '../data/grid'+self.current_time+'.hf'
        xy = h5py.File(grid)
        defect = h5py.File(dgrid)

        xy_times = xy.keys()
        defect_times = defect.keys()
        #print(xy_times)
        #print(defect_times)

        for txy, tdef in zip(xy_times, defect_times):
         #   print('../data/{}.tiff'.format(txy))
            im = np.array(xy.get(txy))
            dgrid = np.array(defect.get(tdef))
            plt.imsave('../data/'+self.current_time+'{}.tiff'.format(txy), schler(im), cmap = 'gray')
            np.savetxt('../data/label_'+self.current_time+'-{}.dat'.format(tdef), dgrid)

        xy.close()
        defect.close()
        
class xyModelFort():
   
    def __init__(self, n, kappa, mu, beta, alpha, plots, it, initScheme, deltime, frame_n = 100, current_time = None):
        ''' 
        Running the fortran simulations in python wrapper

        Args:
            n: size of lattice
            kappa: elastic constant
            mu: field strenght
            beta: 1/Temperature
            alpha: weird landau temperature parameter
            plots: how many snapshots of the simulation to make
            it: how many iterations of the simulation to make
            initScheme: what type of initial conditions (zero, soliton)
            deltime: the time spacing between simulation states
            frame_n: the number of frames to save (if the results are being output to file)

        '''
    

        self.n = n
        self.kappa = kappa
        self.mu = mu
        self.beta = beta
        self.alpha = alpha
        self.plots = plots
        self.it = it
        self.lnoise = np.sqrt(12./beta)
        self.div = int(self.it/self.plots)
        self.initScheme = initScheme
        self.state = np.zeros([self.n,self.n])
        self.deltime = deltime
        self.frame_n = frame_n
        if current_time == None:
            self.current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        else:
            self.current_time = current_time


        
    def initGrid(self, initScheme):
        if (initScheme == 'zero'):
            self.initialState = np.zeros([self.n, self.n], order = 'f')
        elif (initScheme == 'soliton'):
            self.initialState = np.zeros([self.n, self.n],order = 'f')
            langanLib.langan.initializegrid(self.initialState)
            self.initialState = self.solitonInit()
        elif (initScheme == 'random'):
            self.initialState = np.asfortranarray(np.random.rand(self.n,self.n))*2*np.pi
        return self.initialState
    
            
    def setGrid(self, grid):
        self.initialState = grid


            
    def solitonInit(self):
        #mu = self.mu is changed so that we can override this (see initScheme)

        tstart = time.time()
        initMu = .00001
        initIt = 600
        #initialize parameters
        grid = self.initialState
        dgrid = grid
        ngrid = grid
        dh = []


        for t in np.arange(initIt):
            grid = ngrid
            dgrid = 0
            [ngrid,hgrid]=langanLib.langan.update(grid,0.000001,self.kappa, initMu,self.deltime,self.n,self.alpha)
            if(t%(initIt//100)) == 0:

                dh.append([t,np.sqrt(np.mean( (ngrid-grid)**2))])
        dh = np.array(dh).T
        finalState = ngrid
        return finalState
    
    def calc_thermo(self):
        ''' Calculate thermodynamic averages of thermo quantities (magnetization, suscp. etc)'''
        
        averages = 2000
        tstart = time.time()
        #initialize parameters
        dgrid = self.state
        ngrid = self.state
        dh = []
        lgrid = []
        dgrid_list = []
        lham = []
        
        mag = 0.
        xi = 0.
        c = 0.
        
        mag2 = 0.
        xi2 = 0.
        c2 = 0.


        for t in np.arange(averages):
            self.state = ngrid
            dgrid = 0.
            [ngrid,hgrid]=langanLib.langan.update(self.state,self.lnoise,self.kappa, self.mu, self.deltime,self.n,self.alpha)
            magtemp= np.array([np.sum(np.cos(ngrid)),np.sum(np.sin(ngrid))])
            ctemp = self.beta**2*(-ngrid.mean()**2+np.mean(ngrid**2))/self.n
            xitemp = (np.sum(np.cos(ngrid))**2 + np.sum(np.sin(ngrid))**2)/self.n**2
            
            mag += magtemp
            c += ctemp
            xi += xitemp
            
            mag2  += (magtemp)**2
            c2 += (ctemp)**2
            xi2 += (xitemp)**2
            
        self.mag = mag/averages
        self.xi = xi/averages
        self.c = c/averages
        
        self.mag2 =(mag2-mag**2/averages)/(averages-1)
        self.xi2 = (xi2-xi**2/averages)/(averages-1)
        self.c2 = (c2-c**2/averages)/(averages-1)
        duration = time.time() - tstart
        return duration
    
    def simRun(self, savegrid=False, savedefect=False, saveHam = False, grid_n = None, dgrid_n = None, ham_n = None):

        ###File Handling###
        if grid_n == None:
            grid_n = '../data/xy-'+self.current_time+'.hf'
        if dgrid_n == None:
            dgrid_n = '../data/grid'+self.current_time+'.hf'
        if ham_n == None:
            ham_n = '../data/ham'+self.current_time+'.hf'
        twrite = np.unique(np.logspace(0,np.log(self.it),self.frame_n).astype('int'))
        if (savegrid == True):
            xy_hf = h5py.File(grid_n)
        if (savedefect == True):
            defect_hf = h5py.File(dgrid_n)
        if (saveHam == True):
            ham_hf = h5py.File(ham_n)
        tstart = time.time()
        #initialize parameters
        self.state = self.initialState
        dgrid = self.state
        ngrid = self.state
        dh = []
        lgrid = []
        dgrid_list = []
        lham = []


        for t in np.arange(self.it):
            self.state = ngrid
            dgrid = 0
            [ngrid,hgrid]=langanLib.langan.update(self.state,self.lnoise,self.kappa, self.mu,self.deltime,self.n,self.alpha)
            dgrid = langanLib.langan.finddefects(self.state,self.n)
            if(t%(self.it//100)) == 0:

                dh.append([t,np.sqrt(np.mean( (ngrid-self.state)**2))])
            if (t%self.div) == 0:

                lgrid.append(ngrid)
                lham.append(hgrid)
                dgrid_list.append(dgrid)

            ###Writing to File on log time### 
            if t in twrite:
               # print(t)
                #print(twrite)
                if (savegrid == True):
                    xy_hf.create_dataset('t_{}'.format(t), data = ngrid)
                if (savedefect == True):
                    defect_hf.create_dataset('t_{}'.format(t), data = dgrid)
                if (saveHam == True):
                    ham_hf.create_dataset('t_{}'.format(t*self.deltime), data = hgrid)



        dh = np.array(dh).T
        
        #add final values to the end so no simulation time is wasted
        lgrid.append(ngrid)
        lham.append(hgrid)
        dgrid_list.append(dgrid)


        self.snapshots = {}
        self.snapshots['lattice']= lgrid
        self.snapshots['residuals'] = dh
        self.snapshots['defects'] = dgrid_list
        self.snapshots['energy'] = lham
        duration = time.time() - tstart

        ###Close Files, if open ###
        if (savegrid == True):
            xy_hf.close()
        if (savedefect == True):
            defect_hf.close()
        if (saveHam == True):
            ham_hf.close()
        return duration
    def plotRes(self):
        fig,ax = plt.subplots()
        ax.loglog(self.snapshots['residuals'][0],self.snapshots['residuals'][1])
        return [fig,ax]
    
    def plotGrids(self):
        fig,ax = plt.subplots(ncols=2,nrows=4,sharex=True, sharey = True, figsize = (10,10))
        for i,a in enumerate(ax.ravel()):
            #print(i)
            a.imshow(self.snapshots['lattice'][i].T%(2*np.pi), cmap='twilight', vmin=0, vmax=np.pi*2)
            a.set_title('interation {}'.format(i*self.div))
        
        plt.subplots_adjust(wspace=0, hspace=1)
        plt.tight_layout()
        
    def out_defectDensity(self):
        totalD = []
        totalN = []
        for i, frame in enumerate(self.snapshots['defects']):
            posD = np.where(frame > 0)
            negD = np.where(frame < 0)
            ax1.scatter(posD[0],posD[1],i*self.div,c='r', alpha = ka)
            ax1.scatter(negD[0],negD[1],i*self.div,c='b', alpha = ka)
            totalD.append(len(posD[0])/self.n**2)
            totalN.append(len(negD[0])/self.n**2)
        t = np.arange(len(self.snapshots['defects']))*self.div
        
        totalD = [t,totalD]
        totalN = [t,totalN]
        return (totalD, totalN)
    
    def xi_calc(self):
        self.xi = []
        for i, frame in enumerate(self.snapshots['lattice']):
            frame_bound = frame%(2*np.pi)

            M2 = np.sum(np.cos(frame_bound))**2 + np.sum(np.sin(frame_bound))**2
            self.xi.append(M2/self.n**2)
        return self.xi
    def calc_total_mag(self):
        self.mag = []
        for i, frame in enumerate(self.snapshots['lattice']):
            
            M2 = np.array([np.sum(np.cos(frame)),np.sum(np.sin(frame))])
            self.mag.append(M2)
        return self.mag
        
    def c_calc(self):
        self.c = []
        for i, frame in enumerate(self.snapshots['energy']):
            num = -frame.mean()**2+np.mean(frame**2)
            denom = self.n/self.beta**2
            self.c.append(num/denom)
        return self.c

        
    def plotDefects(self):
        fig = plt.figure(figsize = (10,8))
        ax1 = fig.add_subplot(1, 2, 1, projection = '3d')
        ax2 = fig.add_subplot(1,2,2)
        ka = .3
        totalD = []
        totalN = []
        for i, frame in enumerate(self.snapshots['defects']):
            posD = np.where(frame > 0)
            negD = np.where(frame < 0)
            ax1.scatter(posD[0],posD[1],i*self.div,c='r', alpha = ka)
            ax1.scatter(negD[0],negD[1],i*self.div,c='b', alpha = ka)
            totalD.append(len(posD[0])/self.n**2)
            totalN.append(len(negD[0])/self.n**2)
        t = np.arange(len(self.snapshots['defects']))*self.div
        ax2.plot(t,totalD, '.', c = 'r', label = '+ density')
        ax2.plot(t,totalN,'.', c = 'b', label = '- density')
        ax2.legend(loc='best')
        fig.tight_layout()

if __name__ == '__main__':
    t1 = SimData()
    xy = t1.gen()
    t1.gen_training()
