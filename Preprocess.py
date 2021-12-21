import time
import os
import xlrd
import numpy as np 
from pyedflib import highlevel
from google.cloud import storage
from io import StringIO
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d
import re
import sys

class DataPreprocess(object):
    def __init__(self, SLIDE=2, CSA=-1, NOR=0, OSA=1, HYP=2, MSA=3, fs=100, fsDown=4, psg=0, hyp=1, osa=2, csa=3, msa=4, nor=5, lens=6, AHI=7, bucket_name='pranaq_database'):
        #the sliding window has 0.5 sec overlap
        self.SLIDE = SLIDE #/SLIDE(sec)

        #stage assignment
        self.CSA = CSA
        self.NOR = NOR
        self.OSA = OSA
        self.HYP = HYP
        self.MSA = MSA

        #sampling rate / downsampling
        self.fs = fs
        self.fsDown = fsDown
        self.downRate = np.fix(fs/fsDown)

        #state
        self.psg = psg
        self.hyp = hyp
        self.osa = osa
        self.csa = csa
        self.msa = msa
        self.nor = nor
        self.lens = lens
        self.AHI = AHI
        self.bucket_name = bucket_name
        
    def Back2noromal(self, b2n_signal):

        # b2n_signal = loadmat(b2n_signal_mat_path).get('signal').flatten()
        # b2n_signal = [1 , 2, 3, 4, 5, 6, 0 ,0 ,1 ,0 ,1 ,2]

        length = len(b2n_signal)
        zero = "0" 

        for i in range(len(b2n_signal)):
            if b2n_signal[i] == 0:
                zero += "1"
            else:
                zero += "0"

        pattern = re.compile(r'1+')
        slice_part = pattern.findall(zero)
        
        M=[]
        for i in slice_part:
            M.append(len(i))

        find_slice_part_index = re.finditer(r'1+',zero)

        start=[]
        for i in find_slice_part_index:
            start.append(i.start())

        idx = []
        for i in range(len(M)):
            s = max(start[i] - 6*self.fsDown, 1)
            e = min(start[i] + M[i] + 6*self.fsDown, length)+1
            p = range(s,e)
            for j in p:
                idx.append(j)

        idx_uni = np.unique(idx)
        Tem = b2n_signal

        for i in idx_uni:
            Tem[i] = np.nan            # Remove the zeros to prevent the bias of mean

        mu = np.nanmean(Tem)
        for i in idx_uni:
            b2n_signal[i] = mu
        b2n_signal_n = b2n_signal - mu         #Let the mean of signal back to 0.
        return b2n_signal_n

    def prepocessPSG(self, patient_type, name, lightoff_time, startRecord_time):
        recordOffSet = (lightoff_time - startRecord_time)*24*60*60
        THO, ABD, CFlow, SpO2, STAGE, T_LEN = self.InputTestDataLK_PSG(patient_type, name, recordOffSet)
        THO = self.Back2noromal(THO)
        ABD = self.Back2noromal(ABD)
        CFlow = self.Back2noromal(CFlow)
        state = self.InputTestDataLK_Event(patient_type, name, lightoff_time, T_LEN)
        Channels =[THO,ABD,STAGE,CFlow,SpO2]
        return state, Channels

    def InputTestData(self, pnumLK_path, lightoff_path):
        '''
        Input:
            pnumLK_path: str
            lightoff_path: str
        Output:
            pnum: int
            state: 
            Channels: 
        '''
        wb_pnumLK = xlrd.open_workbook(pnumLK_path)
        sheet_pnumLK = wb_pnumLK.sheet_by_index(0)
        pnumLK_df = pd.DataFrame([sheet_pnumLK.row_values(i) for i in range(sheet_pnumLK.nrows )], columns=['type', 'name'])
        
        #open excel file lightoff_path ="./database/LK/lightoff_for_python.xlsx"
        wb_lightoff = xlrd.open_workbook(lightoff_path)
        sheet_lightoff = wb_lightoff.sheet_by_index(0)
        lightoff_df = pd.DataFrame([sheet_lightoff.row_values(i) for i in range(sheet_lightoff.nrows )], columns=['name', 'lightoff_time', 'startRecord_time'])
        
        df = pnumLK_df.merge(lightoff_df, on='name')
        res = df.apply(self.prepocessPSG, axis=1)
        res = pd.DataFrame(res.to_list(), columns=['state', 'Channels'])
        
        pnum = df.shape[0]
        state = res['state'].values.tolist()
        Channels = res['Channels'].values.tolist()
        return pnum, state, Channels

    def trigger_by_cloudfunction(self, patient_type, name, lightoff_time, startRecord_time):
        '''
        Input:
            ticket_path: str
        Output:
            state: 
            Channels: 
        '''
        try:
            return self.prepocessPSG(patient_type, name, lightoff_time, startRecord_time)
        except Exception as err:
            exception_type, exception_object, exception_traceback = sys.exc_info()
            line_number = exception_traceback.tb_lineno
            print("Exception type: ", exception_type)
            print("Error: ", err)
            print("Line number: ", line_number)

    def getEDF(self, bucket_name, blob_name):
        """Get EDF from GCS."""
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        
        blobs = client.list_blobs(bucket_name, prefix=blob_name)
        for b in blobs:
            if b.name.endswith('.edf'):
                blob = b
        blob.download_to_filename('/tmp/tmp.edf')
        PSG_signals, signal_headers, header = highlevel.read_edf('/tmp/tmp.edf')
        os.remove('/tmp/tmp.edf')
        return PSG_signals, signal_headers, header

    def getSTAGE(self, bucket_name, blob_name):
        """Get a STAGE.csv from GCS."""
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        blob = bucket.get_blob(blob_name+'/STAGE.csv')
        blob.download_to_filename('/tmp/STAGE.csv')
        data = pd.read_csv('/tmp/STAGE.csv', encoding= 'unicode_escape')
        os.remove('/tmp/STAGE.csv')
        return data
        # contenido = blob.download_as_string()
        # s = str(contenido,'utf-8')
        # df = pd.read_csv(StringIO(s), encoding= 'unicode_escape')
        # return df

    def InputTestDataLK_PSG(self, ptype, pnum, recordOffSet):
        
        # folder_path ="../../database/LK/"+ ptype +"/"+pnum
        # STAGE_path = folder_path + "/STAGE.csv"
        # items = os.listdir(folder_path)
        # edf_list = []
        # for names in items:
        #     if names.endswith(".edf"):
        #         edf_list.append(names)
        # edf_path = folder_path +"/"+edf_list[0]
        
        # read an edf file
        # PSG_signals, signal_headers, header = highlevel.read_edf(edf_path)
        PSG_signals, signal_headers, header = self.getEDF(self.bucket_name, pnum)

        #'EEG C3-A2', 'EEG C4-A1', 'EEG O1-A2', 'EEG O2-A1', 'EEG A1-A2', 'EOG Left', 'EOG Right', 'EMG Chin', 'ECG I', 'RR', 'ECG II', 'Snore', 'SpO2', 'Flow Patient', 'Flow Patient', 'Effort Tho', 'Effort Abd', 'Body', 'Pleth', 'Leg RLEG', 'Leg LLEG', 'Imp']
        # data = pd.read_csv(STAGE_path, encoding= 'unicode_escape')
        data = self.getSTAGE(self.bucket_name, pnum)

        STAGE = data.values[:,1].tolist()
        CFlow = PSG_signals[13]
        THO = PSG_signals[15]
        ABD = PSG_signals[16]
        SpO2 = PSG_signals[12]
        #SpO2 = [round(num, 2) for num in signals[12]]
        
        STAGE_Len = len(STAGE)  # Total second of this patient
        StartTime = int(np.round(recordOffSet))

        #skip before StartTime*fs & after STAGE_Len*self.fs 
        CFlow = CFlow[StartTime*self.fs:STAGE_Len*self.fs]
        THO = THO[StartTime*self.fs:STAGE_Len*self.fs]
        ABD = ABD[StartTime*self.fs:STAGE_Len*self.fs]
        SpO2 = SpO2[StartTime:STAGE_Len]
        STAGE = STAGE[StartTime:]

        #resample 
        CFlow = signal.resample_poly(CFlow, self.fsDown, self.fs).tolist()
        THO = signal.resample_poly(THO, self.fsDown, self.fs).tolist()
        ABD = signal.resample_poly(ABD, self.fsDown, self.fs).tolist()

        return THO, ABD, CFlow, SpO2, STAGE, STAGE_Len
    
    def StatusRecoder(self, statePSG, stateNOR, state, LENS, time, duration, event):
        for ii, t in enumerate(time):
            start = int(np.fix(t*self.SLIDE))
            end = int(np.fix((t+duration[ii])*self.SLIDE))
            for jj in range(start,end):
                statePSG[jj] = event
                state[jj] = 1
                stateNOR[jj] = 0
            LENS = max(LENS, jj+1)
        return statePSG, stateNOR, state, LENS

    def getXLSX(self, bucket_name, blob_name):
        """Get EDF from GCS."""
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        blob = bucket.get_blob(blob_name+'/Eventlist.xlsx')

        blob.download_to_filename('/tmp/Eventlist.xlsx')
        wb = xlrd.open_workbook('/tmp/Eventlist.xlsx')
        # PSG_signals, signal_headers, header = highlevel.read_edf('Eventlist.xlsx')
        os.remove('/tmp/Eventlist.xlsx')
        return wb

    def InputTestDataLK_Event(self, ptype, pnum, t_lightoff, T_LEN):
        
        Dc, Do, Dm, Dh = [], [], [], []             # duration of each event (record in second)
        t_csa, t_osa, t_msa, t_hyp = [], [], [], [] # start time of each event after light off (record in second)
        numerator = 0

        # STAGE_path ="../../database/LK/"+ ptype +"/"+pnum+"/STAGE.csv"
        # data = pd.read_csv(STAGE_path, encoding= 'unicode_escape')
        data = self.getSTAGE(self.bucket_name, pnum)

        STAGE = data.values.tolist()

        #open excel file
        # wb = xlrd.open_workbook(Eventlist_path)
        wb = self.getXLSX(self.bucket_name, pnum)

        sheet = wb.sheet_by_index(0)
        for index in range(sheet.nrows):
            # if type(sheet.row_values(index)[0]) is not str:
            if not isinstance(sheet.row_values(index)[0], str):
                # Program to extract a particular row value
                time = sheet.row_values(index)[0] # time(convert percentage of day format)
                if time < 0.375:
                    time += 1
                event = sheet.row_values(index)[2] # event(Central apnea/Obstructive apnea....)
                duration = sheet.row_values(index)[3]
                subtration = round((time - t_lightoff)*24*60*60,3)# time - t_lightoff

                if event == "Central apnea":
                    Dc.append(duration)
                    t_csa.append(subtration)
                    numerator += 1
                elif event == "Obstructive apnea":
                    Do.append(duration)
                    t_osa.append(subtration)
                    numerator += 1
                elif event == "Mixed apnea":
                    Dm.append(duration)
                    t_msa.append(subtration)
                    numerator += 1
                elif event == "Hypopnea":
                    Dh.append(duration)
                    t_hyp.append(subtration)
                    numerator += 1

        statePSG = [0]*(T_LEN*self.SLIDE+1)
        stateNOR = [1]*(T_LEN*self.SLIDE+1)  

        LENS = 0   #LEN is the last time that apean event occur
        statePSG, stateNOR, stateOSA, LENS = self.StatusRecoder(statePSG, stateNOR, [0]*(T_LEN*self.SLIDE+1), LENS, t_osa, Do, self.OSA)
        statePSG, stateNOR, stateCSA, LENS = self.StatusRecoder(statePSG, stateNOR, [0]*(T_LEN*self.SLIDE+1), LENS, t_csa, Dc, self.CSA)
        statePSG, stateNOR, stateMSA, LENS = self.StatusRecoder(statePSG, stateNOR, [0]*(T_LEN*self.SLIDE+1), LENS, t_msa, Dm, self.MSA)
        statePSG, stateNOR, stateHYP, LENS = self.StatusRecoder(statePSG, stateNOR, [0]*(T_LEN*self.SLIDE+1), LENS, t_hyp, Dh, self.HYP)
        
        total = 0
        for row in STAGE:
            if row[1] != 11:
                total += 1
        denominator = np.round(total/3600,1)
        return [statePSG[:LENS], stateHYP[:LENS], stateOSA[:LENS], stateCSA[:LENS], stateMSA[:LENS], stateNOR[:LENS], LENS, numerator/denominator]
        ## return = [psg, hyp, osa, csa, msa, nor, lens, AHI]
