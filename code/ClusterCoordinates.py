from sknn.mlp import MultiLayerPerceptron, Layer
import pandas as pd
import numpy as np

class ClusterCoordinates(object):
    def __init__(self, csv_name):
        self.cluster = pd.read_csv(csv_name)
        self.cluster_num = np.ndarray(shape=(len(self.cluster), 2))
        for i in xrange(len(self.cluster)):
            str = ''
            for k in xrange(2,len(self.cluster['COORDINATES'][i])):
                if self.cluster['COORDINATES'][i][k] == ' ':
                    id = k
                    break
                str += self.cluster['COORDINATES'][i][k]
            coordinate_x = float(str)
            str = ''
            for k in xrange(id, len(self.cluster['COORDINATES'][i])):
                if self.cluster['COORDINATES'][i][k] == ' ':
                    continue
                if self.cluster['COORDINATES'][i][k] == ']':
                    break
                str += self.cluster['COORDINATES'][i][k]
            coordinate_y = float(str)
            self.cluster_num[i][0] = coordinate_x
            self.cluster_num[i][1] = coordinate_y
        self.cluster_num = np.transpose(self.cluster_num)
    def getCoordinates(self):
        return self.cluster_num