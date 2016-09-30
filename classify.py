'''
@Author: Brandon Radosevich
@Date:  September 24, 2016
@Description: Classifies a given folder of songs based on the training model.
'''
from __future__ import division # for printing progress
import sys
import cPickle
import os
import argparse
from songTraining import SongTraining
import htk_reader # for reading HTK From file
from htk_reader import HTKFeat_read
from sklearn.mixture import GMM
import numpy as np
import glob
import re
_author__ = "Brandon Radosevich <bradosev@nmsu.edu>"
__file__ = "classify.py"

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts
'''
Class: classifySong
Date: September 25, 2016
Description: This class is used for classifying any given number of songs according to a training file
created from songTraining.py
'''
class classifySong(object):

    def __init__(self,trainingFile,songData,folder,filename, components, iterations):
        self.components = 1
        self.iterations = 1
        if components != None and iterations != None:
            self.components = components
            self.iterations = iterations
        self.filename = filename
        self.predictedSongs = {}
        self.vectorLength = 20
        training = self.readTrainingFile(trainingFile)
        self.model = training
        self.songs = []
        if songData == None:
            self.readFilesFromFolder(folder)
        else:
            self.readTestData(songData,folder)

    '''
    function: readTrainingFile
    date: September 25, 2016
    description: This function reads in the training model from a file and prepares it for classifying.
    '''
    def readTrainingFile(self,filename):
        with open(filename,'rb') as f:
            gmm_training = cPickle.load(f)
        return gmm_training

    '''
    function: readFilesfromFolder
    date: September 25, 2016
    description: If no key file is provided, the code performs a ls -l on a given folder to find the songs.
    '''
    def readFilesFromFolder(self,folder):
        length = len(glob.glob(folder+"/*.htk"))
        count = 0
        for filename in sorted(glob.glob(folder+'/*.htk')):
            self.classify(filename)
            count = count + 1
            sys.stdout.write('\r')
            sys.stdout.write('%.2f%% Complete ' % ( count / length * 100,))
            sys.stdout.flush()
        print "\n"
        self.writeToFile()
    '''
    function: readTestData
    date: September 25, 2016
    description: Reads in the path to all of the song vectors to test. It then classifies them, one by one.
    '''
    def readTestData(self,songData,folder):
        kData = [line.strip("\r\n") for line in open(songData)]
        count = 0
        song_keys = []
        self.verbose = 0
        for f in kData:
            s,k = f.split(",")
            self.classify(folder+"/"+s)
            count = count + 1
            sys.stdout.write('\r')
            sys.stdout.write('%.2f%% Complete ' % ( count / len(kData) * 100,))
            sys.stdout.flush()
        print "\n"
        self.writeToFile()
    '''
    function: HTKRead
    date: September 25, 2016
    description: This function is meant to be called from other classes.
    '''
    def htkRead(self,songName):
        a = htk_reader.open(songName,'r',self.vectorLength)
        self.verbose = 0
        g = self.getGMMValues(a.getall())
        return g
    '''
    function: htkGetVector
    date: September 25, 2016
    description: This function gets the frequency vector from a given file.
    '''
    def htkGetVector(self,songName):
        a = htk_reader.open(songName,'r',self.vectorLength)
        return a.getall()

    '''
    function: getGMMValues
    date: September 25, 2016
    description: This function finds the GMM for a given song Vector.
    '''
    def getGMMValues(self,sVector):
        g = GMM(n_components=self.components,covariance_type='diag',init_params='wmc',n_iter=self.iterations,params='wmc',verbose=self.verbose)
        g.fit(sVector)
        return g

    '''
    function: classify
    date: September 25, 2016
    description: This function classifies a song by comparing the gmm to the given vector.
    '''
    def classify(self,songPath):
        vector = self.htkGetVector(songPath)
        ml_songTitle = ''
        max_likelihood = -9e99
        for title, gmm in self.model.items():
            likelihood = gmm.score_samples(vector)[0]
            likelihood = np.sum(likelihood)
            if max_likelihood < likelihood:
                max_likelihood = likelihood
                ml_songTitle = title
        self.songs.append(ml_songTitle)

    '''
    function: writeToFile
    date: September 25, 2016
    description: This functino writes the predicted results to a text file.
    '''
    def writeToFile(self):
        w = open(self.filename,"w")
        for s in self.songs:
            w.write(s + "\n")
        w.close

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Spam Filter Training")
    parser.add_argument("-t","--trainingFile", dest="trainingFile", help="The training model to use", metavar="FILE",required=True)
    parser.add_argument("-s","--key",dest="songData",help="The Key file for training", metavar="FILE",required=False)
    parser.add_argument("-f","--folder",dest="folder",help="Folder for testing", metavar="FOLDER",required=True)
    parser.add_argument("-fN","--fileName",dest="filename",help="FileName to Save To", metavar="FILE",required=True)
    parser.add_argument("-K","--components",dest="components",help="Number Of Components", metavar="FILE",required=False)
    parser.add_argument("-I","--iterations",dest="iterations",help="Number Of Iterations to Use", metavar="FILE",required=False)
    args = parser.parse_args()
    if args.iterations != None and args.components != None:
        sp = classifySong(args.trainingFile,args.songData, args.folder,args.filename, args.components, args.iterations)
    elif args.iterations == None and args.components == None:
        sp = classifySong(args.trainingFile,args.songData, args.folder, args.filename,None, None)
    elif args.keyFile == None:
        sp = classifySong(args.trainingFile, None, args.folder, args.filename, None, None)
    else:
        print "Usage: "
        print " python classify.py -t training.pcl -f CompanionFiles3/data/test1/ -s CompanionFiles3/data/test1.key -fN predictions1.txt"
        print " python classify.py -t training.pcl -f CompanionFiles3/data/test1/ -k CompanionFiles3/data/test1.key -fN predictions1.txt -K 20 -I 10"
