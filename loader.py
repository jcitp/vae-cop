from __future__ import division
import numpy as np
import pandas
import glob

def readDirectory(directory,temperatures,temperatureRepetitions,samplesPerFile,labels,size):
    
    samples=temperatureRepetitions*temperatures*samplesPerFile
    alldata=np.zeros((samples,size+labels),dtype=float)
    filelist=sorted(glob.glob(directory+"/*gz"))
    print("Loading %s files."%len(filelist))
    for i,file in enumerate(filelist):
        data=pandas.read_table(file,sep=" ",skiprows=[0],nrows=samplesPerFile,header=None)
        alldata[i*samplesPerFile:(i+1)*samplesPerFile]=data.values[:,:(size+labels)]
        
    data=alldata[:,:size]
    labels=alldata[:,size:]
    return samples,data,labels

def readFile(file,size):
    
    alldata=pandas.read_table(file,sep=" ",skiprows=[0],header=None).values
    data=np.zeros((alldata.shape[0],int(np.sqrt(size/2)),2*int(np.sqrt(size/2)),1))
    data[:,:,1::2]=alldata[:, :size:2].reshape((alldata.shape[0],int(np.sqrt(size/2)),int(np.sqrt(size/2)),1))
    data[:,:, ::2]=alldata[:,1:size:2].reshape((alldata.shape[0],int(np.sqrt(size/2)),int(np.sqrt(size/2)),1))
    labels=alldata[:,size:]
    return alldata.shape[0],data,labels
    