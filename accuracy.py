'''
@Author: Brandon Radosevich
@Date:  September 26, 2016
@Description: Finds the accuracy of the predictions based on a given key.
'''
import argparse
'''
class: Accuracy
date: September 26, 2016
definition: This class finds the accuracy of the predictions with a given key.
'''
class SongAccuracy(object):

    def __init__(self,predictions, actual):
        actual_keys = self.getKeys(actual,1)
        predicted_keys = self.getKeys(predictions)
        self.calculateAccuracy(predicted_keys,actual_keys)

    '''
    function: getKeys
    date: September 26, 2016
    Description: This function gets the keys from a given file and places them in an array.
    '''
    def getKeys(self,actual,place=0):
        kData = [line.strip("\r\n") for line in open(actual)]
        keys = []
        for k in kData:
            key = k.split(",")
            keys.append(key[place])
        return keys

    '''
    function: calculateAccuracy
    date: September 26, 2016
    Description:This function finds the accuracy of the predictions keys vs the actual keys.
    '''
    def calculateAccuracy(self,predictions, actual):
        count = 0
        for p,a in zip(predictions,actual):
            #print p,a
            if p in a:
                count = count + 1
        accuracy = float(count) / float(len(actual)) * 100
        print 'Accuracy is: %0.2f %%'% (accuracy,)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Spam Filter Training")
    parser.add_argument("-m","--modelFile", dest="predictions", help="The file to read data from", metavar="FILE",required=True)
    parser.add_argument("-k","--key",dest="actualKey",help="The Key file for Accuracy", metavar="FILE",required=True)
    parser.add_argument("-f","--folder",dest="folder",help="Folder for testing", metavar="FILE",required=False)
    args = parser.parse_args()
    sp = SongAccuracy(args.predictions,args.actualKey)
