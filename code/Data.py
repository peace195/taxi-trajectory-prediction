import json
import zipfile
import datetime
import numpy as np
import pandas as pd

class Data:
    lookupTable = {}

    def __init__(self, zip_name, csv_name, 
            n_begin_end, origin_call, origin_stand,
            week, day, qhour, day_type, taxi_id):
        self.zf = zipfile.ZipFile(zip_name)
        self.df = pd.read_csv(self.zf.open(csv_name), nrows = 250,
                                converters = {'POLYLINE': lambda x: json.loads(x)})

        self.n_begin_end    = n_begin_end 
        self.dim_embeddings = [
            ('origin_call', 57106, origin_call),
            ('origin_stand', 64, origin_stand),
            ('week_of_year', 52, week),
            ('day_of_week', 7, day),
            ('qhour_of_day', 24 * 4, qhour),
            ('day_type', 3, day_type),
            ('taxi_id', 448, taxi_id),
        ]
        self.origin_call = origin_call
        self.origin_stand = origin_stand
        self.week = week
        self.day = day
        self.qhour = qhour
        self.day_type = day_type
        self.taxi_id = taxi_id
        self.dim_input = n_begin_end*2*2 + sum(x for (_, _, x) in self.dim_embeddings)

    def outputs(self):
        return np.matrix([[p[len(p)-1][0], p[len(p)-1][1]] 
            for p in self.df['POLYLINE'] if len(p) > self.n_begin_end])

    def inputsPos(self):
        n_begin_end_pos = np.matrix([[p[i-1][0], p[i-1][1], p[len(p)-i][0], p[len(p)-i][1]] 
            for p in self.df['POLYLINE'] 
            for i in range(1, self.n_begin_end + 1)
            if len(p) > self.n_begin_end]) 
        n_begin_end_pos = np.reshape(n_begin_end_pos,
                (n_begin_end_pos.shape[0]/self.n_begin_end, self.n_begin_end*2*2))
        return n_begin_end_pos

    def buildTable(self):
        sub_dict = {str(el):np.random.rand(self.origin_call)
            for el in pd.unique(self.df['ORIGIN_CALL'])}
        Data.lookupTable['origin_call'] = sub_dict 

        sub_dict = {str(el):np.random.rand(self.origin_stand)
            for el in pd.unique(self.df['ORIGIN_STAND'])}
        Data.lookupTable['origin_stand'] = sub_dict 

        sub_dict = {str(el):np.random.rand(self.taxi_id)
            for el in pd.unique(self.df['TAXI_ID'])}
        Data.lookupTable['taxi_id'] = sub_dict 

        sub_dict = {str(el):np.random.rand(self.day_type)
            for el in pd.unique(self.df['DAY_TYPE'])}
        Data.lookupTable['day_type'] = sub_dict 

        sub_dict = {str(el):np.random.rand(self.week)
            for el in range(1,54)}
        Data.lookupTable['week_of_year'] = sub_dict 

        sub_dict = {str(el):np.random.rand(self.day)
            for el in range(1,8)}
        Data.lookupTable['day_of_week'] = sub_dict 

        sub_dict = {str(el):np.random.rand(self.qhour)
            for el in range(1,100)}
        Data.lookupTable['qhour_of_day'] = sub_dict 

    def inputs(self):
        self.buildTable()
        inputs = self.inputsPos()
        n_samples = self.df.shape[0]
        append = np.matrix([Data.lookupTable['origin_call'][str(self.df['ORIGIN_CALL'][i])]
            for i in range (0, n_samples)
            if len(self.df['POLYLINE'][i]) > self.n_begin_end])
        inputs = np.concatenate((inputs, append), axis = 1)
        append = np.matrix([Data.lookupTable['origin_stand'][str(self.df['ORIGIN_STAND'][i])]
            for i in range (0, n_samples)
            if len(self.df['POLYLINE'][i]) > self.n_begin_end])
        inputs = np.concatenate((inputs, append), axis = 1)
        append = np.matrix([Data.lookupTable['taxi_id'][str(self.df['TAXI_ID'][i])]
            for i in range (0, n_samples)
            if len(self.df['POLYLINE'][i]) > self.n_begin_end])
        inputs = np.concatenate((inputs, append), axis = 1)
        append = np.matrix([Data.lookupTable['day_type'][str(self.df['DAY_TYPE'][i])]
            for i in range (0, n_samples)
            if len(self.df['POLYLINE'][i]) > self.n_begin_end])
        inputs = np.concatenate((inputs, append), axis = 1)
        append = np.matrix([Data.lookupTable['week_of_year']
            [str(datetime.datetime.fromtimestamp(int(self.df['TIMESTAMP'][i])).
                isocalendar()[1])]
            for i in range (0, n_samples)
            if len(self.df['POLYLINE'][i]) > self.n_begin_end])
        inputs = np.concatenate((inputs, append), axis = 1)
        append = np.matrix([Data.lookupTable['day_of_week']
            [str(datetime.datetime.fromtimestamp(int(self.df['TIMESTAMP'][i])).
                isocalendar()[2])]
            for i in range (0, n_samples)
            if len(self.df['POLYLINE'][i]) > self.n_begin_end])
        inputs = np.concatenate((inputs, append), axis = 1)
        append = np.matrix([Data.lookupTable['qhour_of_day']
           [str(datetime.datetime.fromtimestamp(int(self.df['TIMESTAMP'][i])).time().hour*4 +
           datetime.datetime.fromtimestamp(int(self.df['TIMESTAMP'][i])).time().minute/15 + 1)]
            for i in range (0, n_samples)
            if len(self.df['POLYLINE'][i]) > self.n_begin_end])
        inputs = np.concatenate((inputs, append), axis = 1)
        
        return inputs

#from Data import Data
#data    = Data('../train.csv.zip', 'train.csv', 5, 10, 10, 10, 10, 10, 10, 10)
#outputs = data.outputs()
#inputs  = data.inputs()
