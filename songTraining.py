'''
@Author: Brandon Radosevich
@Date: September 19, 2016
@Description: songTraining
'''
#Import Statements
from __future__ import division # for printing progress
import sys # for printing progress
import argparse # for ArgumentParser
import os   # for listing director
import numpy as np #for numpy
import sklearn # for GMM
from sklearn.mixture import GMM
import glob # for reading files in folder
import htk_reader # for reading HTK From file
from htk_reader import HTKFeat_read
import csv # for writing to file
from itertools import izip_longest
import cPickle # for dumping training model to .pcl file
import subprocess
__author__ = "Brandon Radosevich <bradosev@nmsu.edu>"
__file__ = "songTraining.py"

'''
Class: SongTraining
Date: September 25, 2016
Description: This class is used for training a model for predicting songs.
'''
class SongTraining(object):

    def __init__(self,folder, keyFile, filename, components, iterations):
        self.components = 1
        self.iterations = 1
        if components != None and iterations != None:
            self.components = components
            self.iterations = iterations
        self.filename = filename
        self.vectorLength = 20
        self.verbose = 1
        self.trainingDict = {}
        self.readFolder(folder,keyFile)

    '''
    function: readFolder
    date: September 19, 2016
    description: This function reads in a folder and keyFile with the paths to each folder for training.
    It then iterates through all files and performs a GMM model.
    '''
    def readFolder(self,folder,keyFile):
        kData = [line.strip("\r\n") for line in open(keyFile)]
        count = 0
        self.verbose = 0
        paths = []
        songs = []
        for f in kData:
            s,k = f.split(",")
            self.HTKRead(folder+"/"+s,k)
            count = count + 1
            sys.stdout.write('\r')
            sys.stdout.write('%.2f%% Complete ' % ( count / len(kData) * 100,))
            sys.stdout.flush()
        self.saveTraining()

    '''
    function: HTKRead
    date: September 20, 2016
    description: This function determines the song vector for a given file.
    '''
    def HTKRead(self,songName,keyFile):
        a = htk_reader.open(songName,'r',self.vectorLength)
        self.getGMMValues(a.getall(),keyFile)

    '''
    function: getGMMValues
    date: September 20, 2016
    description: This function finds the appropriate GMM for a given song.
    '''
    def getGMMValues(self,sVector,songName):
        self.trainingDict[songName] = GMM(n_components=self.components,covariance_type='diag',init_params='wmc',n_iter=self.iterations,params='wmc',verbose=self.verbose)
        self.trainingDict[songName].fit(sVector)

    '''
    function: saveTraining
    date: September 22, 2016
    description: This function saves the entire data model to a file for use in classification.
    '''
    def saveTraining(self):
        with open(self.filename,"wb") as f:
            cPickle.dump(self.trainingDict, f)
        print "\nSaved Training to %s" %(self.filename,)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Spam Filter Training")
    parser.add_argument("-s","--songsFolder", dest="folder", help="The file to read data from", metavar="PATH",required=True)
    parser.add_argument("-k","--key",dest="keyFile",help="The Key file for training", metavar="FILE",required=True)
    parser.add_argument("-fN","--fileName",dest="filename",help="FileName to Save To", metavar="FILE",required=True)
    parser.add_argument("-K","--components",dest="components",help="Number Of Components", metavar="FILE",required=False)
    parser.add_argument("-I","--iterations",dest="iterations",help="Number Of Iterations to Use", metavar="FILE",required=False)
    args = parser.parse_args()
    if args.iterations != None and args.components != None:
        sp = SongTraining(args.folder,args.keyFile,args.filename, args.components, args.iterations)
    elif args.iterations == None and args.components == None:
        sp = SongTraining(args.folder,args.keyFile,args.filename,None, None)
    else:
        print "Usage: "
        print " python songTraining.py -s CompanionFiles3/data/train/ -k CompanionFiles3/data/train.key -fN training.pcl"
        print " python songTraining.py -s CompanionFiles3/data/train/ -k CompanionFiles3/data/train.key -fN training.pcl -K 20 -I 10"
