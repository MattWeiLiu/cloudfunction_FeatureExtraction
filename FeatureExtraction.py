import time
import math
import numpy as np 
import pandas as pd
from scipy.io import loadmat
from numba import jit
from scipy import signal
import threading
from queue import Queue
import multiprocessing

class FeatureExtraction(object):
    def __init__(self, SLIDE=2, NOR=0, fs=100, fsDown=4, psg=0, nor=5, lens=6, CurrentWindowAR=10, PreviousWindowAR=60, 
        CurrentWindowFR=10, CurrentWindowCov=10, QtAR=95, SpO2Windowlen=40, SpO2delaylen=40, eps=0.00000001):
        # the sliding window has 0.5 sec overlap
        self.SLIDE = SLIDE #/SLIDE(sec)

        #stage assignment
        self.NOR = NOR

        #sampling rate / downsampling
        self.fs = fs
        self.fsDown = fsDown

        #state
        self.psg = psg
        self.lens = lens

        #window length for AR (CW and PW) unit: sec
        self.CurrentWindowAR = CurrentWindowAR
        self.PreviousWindowAR = PreviousWindowAR

		#window lenth for FR. unit: sec
        self.CurrentWindowFR = CurrentWindowFR

		#window length for Cov. unit: sec
        self.CurrentWindowCov = CurrentWindowCov

        self.CurrentWindowARlen = self.CurrentWindowAR * self.SLIDE
        self.PreviousWindowARlen = self.PreviousWindowAR * self.SLIDE
        self.CurrentWindowFRlen = self.CurrentWindowFR * self.SLIDE
		
        self.CurrentWindowCovlen = self.CurrentWindowCov * self.SLIDE

        self.QtAR = QtAR
        self.SpO2Windowlen = SpO2Windowlen
        self.SpO2delaylen = SpO2delaylen
        self.eps = eps

    def smooth(self, signal, Fs, func): 
        # reproduce matlab moving funcion (eg: movmean, movsum) via Pandas
        Fs = int(Fs)
        if func == 'mean':
            return pd.DataFrame(signal).rolling(window=Fs, min_periods=1, center=True).mean()[0].values
        elif func == 'sum':
            return pd.DataFrame(signal).rolling(window=Fs, min_periods=1, center=True).sum()[0].values
        else:
            print ('Parameter "func" is required')

    def One_stage_AR(self, THO, ABD, standWinidx, previousWinidx, currentWinidx, stg, statePSG):
        preTHO, preABD, curTHO, curABD, stdTHO, stdABD = [], [], [], [], [], []

        if stg == 11:
            AR_THO = float("nan")
            AR_ABD = float("nan")
        else:
            if statePSG == self.NOR:
                for ii in previousWinidx:
                    preTHO.append(THO[ii])
                    preABD.append(ABD[ii])
                for jj in currentWinidx:
                    curTHO.append(THO[jj])
                    curABD.append(ABD[jj])

                AR_THO = np.percentile(np.absolute(curTHO), self.QtAR, interpolation='midpoint')/np.percentile(np.absolute(preTHO), self.QtAR, interpolation='midpoint')
                AR_ABD = np.percentile(np.absolute(curABD), self.QtAR, interpolation='midpoint')/np.percentile(np.absolute(preABD), self.QtAR, interpolation='midpoint')
            else:
                for ii in standWinidx:
                    stdTHO.append(THO[ii])
                    stdABD.append(ABD[ii])
                for jj in currentWinidx:
                    curTHO.append(THO[jj])
                    curABD.append(ABD[jj])
               
                AR_THO = np.percentile(np.absolute(curTHO), self.QtAR, interpolation='midpoint')/np.percentile(np.absolute(stdTHO), self.QtAR, interpolation='midpoint')
                AR_ABD = np.percentile(np.absolute(curABD), self.QtAR, interpolation='midpoint')/np.percentile(np.absolute(stdABD), self.QtAR, interpolation='midpoint')

        return AR_THO, AR_ABD

    @jit
    def CurveExt_M_sub_jit(self, FVal, E, m, n, lam):
        '''
        Input:
            FVal: ndarray
            E: ndarray
            m: int
            n: int
            lam: int
        Output:
            Curve: list
        '''
        # print (m, n)
        for ii in range(1, m): #time
            for jj in range(n): #freq
                for kk in range(n):#calculate the penalty term #
                    FVal[ii][jj] = min(FVal[ii][jj], FVal[ii-1][kk]+lam*np.square((kk-jj))) 
                #E(ii,jj) is the SST value at time ii and freq jj
                FVal[ii][jj] = FVal[ii][jj] + E[ii][jj]

        Curve = [0]*m     
        # Curve = np.zeros(m)
        Curve[m-1] = int(np.argmin(FVal[m-1]))+1
        for ii in range(m-2,-1,-1):
            a = FVal[ii+1][Curve[ii+1]-1] 
            b = E[ii+1][Curve[ii+1]-1]
            val = a - b 
            for kk in range(n):
                if (abs(val -FVal[ii][kk] -lam*np.square(kk+1-Curve[ii+1]))< self.eps):
                    Curve[ii] = kk+1
                    break
            if Curve[ii] == 0:
                Curve[ii] = int(np.round(n/2))
        return Curve

    def CurveExt_M_jit(self, E, lam):  
        '''
        Input:
            E: ndarray
            lam: int
        Output:
            Curve: list
        '''
        # E_sum = np.sum(E)
        E_sum = E.sum()
        E /= E_sum
        E = -np.log(E+self.eps)
        m, n = E.shape
        FVal = np.full((m, n), float("inf")) # m is time, n is freq
        # FVal[0] = E[0,:]
        FVal[0] = E[0]
        Curve = self.CurveExt_M_sub_jit(FVal, E, m, n, lam)
        return Curve

    def CurveExt_M(self, E, lam): 
        E_sum = np.sum(E)
        E = E/E_sum
        E = -np.log(E+self.eps)
        m,n = np.shape(E)
        FVal = np.array([[float("inf")]*n]*m) #m is time, n is freq
        FVal[0] =E[0,:]
        Curve = [0]*m
        for ii in range(1,m): #time
            for jj in range(n): #freq
                for kk in range(n):#calculate the penalty term #
                    FVal[ii][jj] = min(FVal[ii][jj], FVal[ii-1][kk]+lam*np.square((kk-jj))) 
                #E(ii,jj) is the SST value at time ii and freq jj
                FVal[ii][jj] = FVal[ii][jj] + E[ii][jj]

        Curve = [0]*m     
        Curve[m-1] = int(np.argmin(FVal[m-1]))+1
        for ii in range(m-2,-1,-1):
            a = FVal[ii+1][Curve[ii+1]-1] 
            b = E[ii+1][Curve[ii+1]-1]
            val = a - b 
            for kk in range(n):
                if (abs(val -FVal[ii][kk] -lam*np.square(kk+1-Curve[ii+1]))< self.eps):
                    Curve[ii] = kk+1
                    break
            if Curve[ii] == 0:
                Curve[ii] = int(np.round(n/2))
        return Curve

    def hermf(self, N, M, tm):
    
        dt = 2*tm/(N-1)
        tt = np.linspace(-tm,tm,N)
        g = np.exp(-np.square(tt)/2) 
        P = [np.ones(N)] 
        P = np.append(P, [2*tt], axis=0)

        for k in range(2,M+1):
            P = np.append(P,[2*np.multiply(tt,P[k-1]) - 2*(k-1)*P[k-2]],axis=0) 
         
        Htemp = [np.multiply(P[0],g)/math.sqrt(math.sqrt(math.pi)*math.gamma(1))*math.sqrt(dt)] 
        for k in range(1,M+1): 
            Htemp = np.append(Htemp, [np.multiply(P[k],g)/math.sqrt(math.sqrt(math.pi)
                               *math.pow(2,k)*math.gamma(k+1))*math.sqrt(dt)],axis = 0); 
           

        h = Htemp[0:M,:];          
        Dh = [(np.multiply(tt,Htemp[0]) - math.sqrt(2)*Htemp[1])*dt] 
        for k in range(1,M): 
            Dh = np.append(Dh,[(np.multiply(tt,Htemp[k]) - math.sqrt(2*(k+1))*Htemp[k+1])*dt] ,axis=0)
        
        Dh = np.array(Dh)
        return h,Dh,tt

    def sqSTFTbase2nd(self, x, lowFreq, highFreq, alpha, tDS, h, Dh, DDh, online):
        x=np.array([x])
        xrow,xcol = x.shape
        t = np.arange(1,xcol+1)
        tLen = len(np.arange(1,xcol+1,tDS))

        """ for tfr """
        N = len(np.arange(-0.5+alpha,0.5,alpha))+1
        """ for tfrsq """
        Lidx = int(np.ceil( (N/2)*(lowFreq/0.5) )+1) 
        Hidx = int(np.floor( (N/2)*(highFreq/0.5)))
        fLen = Hidx - Lidx + 1
        """===================================================================="""
        """check input signals"""
        if (xrow!=1):
            raise ValueError('X must have only one row')
        elif (highFreq > 0.5):
            raise ValueError('TopFreq must be a value in [0, 0.5]')
        elif (tDS < 1) or (tDS%1): 
            raise ValueError('tDS must be an integer value >= 1')

        hrow,hcol = h.shape
        Lh = int((hcol-1)/2) 
        if ((hrow!=1) or ((hcol%2)==0)):
            raise ValueError('H must be a smoothing window with odd length')

        ht = np.arange(-Lh,Lh+1)
        """===================================================================="""
        """run STFT and reassignment rule"""
        if online: 
            tfr = np.zeros((100, int(N/2)),dtype=complex)     
            """ for h"""
            tfrsq = np.zeros((100, fLen),dtype=complex)
            tfrsq2nd = np.zeros((100, fLen),dtype=complex)
        else: 
            tfr = np.zeros((tLen, int(N/2)),dtype=complex) 
            """for h"""
            tfrsq = np.zeros((tLen, fLen),dtype=complex)
            tfrsq2nd = np.zeros((tLen, fLen),dtype=complex) 

        tfrtic = np.linspace(0, 0.5, N/2)
        tfrsqtic = np.linspace(lowFreq, highFreq, fLen)
        
        """Ex = mean(abs(x(min(t):max(t))).^2);"""
        Ex = np.mean(np.square(np.absolute(x[0,np.amin(t):np.amax(t)+1])))
        Threshold = math.pow(10,-8)*Ex  
        """% originally it was 1e-6*Ex"""

        for tidx in range(1,tLen+1):
            """ti is the current time"""
            ti = t[(tidx-1)*tDS]
            """tau is the relevant index associated with ti"""
            tau = (np.arange(-np.amin([round(N/2)-1,Lh,ti-1]),np.amin([round(N/2)-1,Lh,xcol-ti])+1)).astype(int)
            """indices is the absolute index in the "evaluation window" """
            indices= ((N+tau)%N+1).astype(int)
            norm_h=np.linalg.norm(h[:,(Lh+tau).astype(int)])
            tf0 = np.zeros((1, N),dtype=complex)
            tf1 = np.zeros((1, N),dtype=complex)
            tf2 = np.zeros((1, N),dtype=complex)
            tfx0 = np.zeros((1, N),dtype=complex)
            tfx1 = np.zeros((1, N),dtype=complex)
            tf0[:,indices-1] = x[:,ti+tau-1]*np.conjugate(h[:,Lh+tau])/norm_h
            tf1[:,indices-1] = x[:,ti+tau-1]*np.conjugate(Dh[:,Lh+tau])/norm_h
            tf2[:,indices-1] = x[:,ti+tau-1]*np.conjugate(DDh[:,Lh+tau])/norm_h
            tfx0[:,indices-1] = x[:,ti+tau-1]*np.conjugate(h[:,Lh+tau])*ht[Lh+tau]/norm_h
            tfx1[:,indices-1] = x[:,ti+tau-1]*np.conjugate(Dh[:,Lh+tau])*ht[Lh+tau]/norm_h
            tf0 = np.fft.fft(tf0);tf0 = tf0[:,0:int(N/2)]
            tf1 = np.fft.fft(tf1);tf1 = tf1[:,0:int(N/2)]
            tf2 = np.fft.fft(tf2);tf2 = tf2[:,0:int(N/2)]
            tfx0 = np.fft.fft(tfx0);tfx0 = tfx0[:,0:int(N/2)]
            tfx1 = np.fft.fft(tfx1);tfx1 = tfx1[:,0:int(N/2)]   
            """% get the first order omega"""
            omega = np.round(N*np.imag(tf1/tf0)/(2.0*math.pi))
            """% get the 2nd order omega"""
            omega2nd = np.round(N * np.imag(tf1/tf0 - (tf0*tf2-tf1*tf1)/(tfx1*tf0-tfx0*tf1)*tfx0/tf0)/(2.0*math.pi))
            
            
            sst = np.zeros((1,fLen),dtype=complex)
            sst2nd = np.zeros((1,fLen),dtype=complex)
            
            for jcol in range(1,int(N/2)+1):
                if abs(tfr[0,jcol-1]) > Threshold:
                    
                    jcolhat = int(jcol - omega[:,jcol-1])
                    jcolhat2nd = int(jcol - omega2nd[:,jcol-1])
                    
                    if (jcolhat < Hidx+1) and (jcolhat >= Lidx):
                        sst[0,jcolhat-Lidx] = sst[0,jcolhat-Lidx]+tf0[0,jcol-1]
                    if (jcolhat2nd < Hidx+1) and (jcolhat2nd >= Lidx):
                        sst2nd[0,jcolhat2nd-Lidx] = sst2nd[0,jcolhat2nd-Lidx]+tf0[0,jcol-1]
                    
                    
            if online:
                tfr[0:99,:] = tfr[1:100,:]
                tfrsq[0:99,:] = tfrsq[1:100,:]
                tfrsq2nd[0:99,:] = tfrsq2nd[1:100,:]
                tfr[99,:] = tf0[0,0:int(N/2)]
                tfrsq[99,:] = sst
                tfrsq2nd[99,:] = sst2nd
            
            else:
                tfr[tidx-1,:] = tf0[0,0:int(N/2)]
                tfrsq[tidx-1,:] = sst
                tfrsq2nd[tidx-1,:] = sst2nd

        return np.transpose(tfr), tfrtic, np.transpose(tfrsq), np.transpose(tfrsq2nd), tfrsqtic


    def sqSTFTbase(self, x, lowFreq, highFreq, alpha, tDS, h, Dh, Smooth, Hemi):
        '''
        Input:
            x: ndarray
            lowFreq: int
            highFreq: float
            alpha: float
            tDS: int
            h: ndarray
            Dh: ndarray
            Smooth: int
            Hemi: int
        Output:
            tfr: ndarray
            tfrtic: ndarray
            tfrsq: ndarray
            tfrsqtic: ndarray
        '''

        x = x.reshape(1, -1)
        xrow, xcol = x.shape
        t = np.arange(1, xcol+1)
        tidxs = np.arange(1, xcol+1, tDS)
        tLen = len(tidxs) # tLen length: 87368

        """ for tfr """
        N = len(np.arange(-0.5+alpha,0.5,alpha))+1
        """ for tfrsq """
        Lidx = round( (N/2)*(lowFreq/0.5) )+1 
        Hidx = round( (N/2)*(highFreq/0.5))
        fLen = Hidx - Lidx + 1

        "=========================================================================="
        "check input signals"
       
        if (xrow!=1):
            raise ValueError('X must have only one row')
        elif (highFreq > 0.5):
            raise ValueError('TopFreq must be a value in [0, 0.5]')
        elif (tDS < 1) or (tDS%1): 
            raise ValueError('tDS must be an integer value >= 1')
         
        hrow, hcol = h.shape
        Lh = int((hcol-1)/2) 
        if ((hrow!=1) or ((hcol%2)==0)):
            raise ValueError('H must be a smoothing window with odd length')

        "=========================================================================="
        "run STFT and reassignment rule"
        tfr = np.zeros((tLen, int(N/2)), dtype=complex)
        tfrtic = np.linspace(0, 0.5, int(N/2)) 
        tfrsq = np.zeros((tLen, fLen), dtype=complex)
        tfrsqtic = np.linspace(lowFreq, highFreq, fLen)


        """Ex = mean(abs(x(min(t):max(t))).^2);"""
        Ex = np.mean(np.square(np.absolute(x)))
        Threshold = math.pow(10, -8)*Ex  
        """% originally it was 1e-6*Ex"""

        Mid = round(len(tfrsqtic)/2)
        Delta = 20*np.square(tfrsqtic[1]-tfrsqtic[0]) 
        weight = np.exp(-np.square(tfrsqtic[Mid-11:Mid+10]-tfrsqtic[Mid-1])/Delta)
        weight = weight/np.sum(weight)
        weightIDX = np.arange(Mid-10,Mid+11,1) - Mid
        #for tidx in range(1):
        for Idx, tidx in enumerate(tidxs):
            ti = t[tidx-1]
            tau = (np.arange(-np.amin([round(N/2)-1, Lh, ti-1]), np.amin([round(N/2)-1, Lh, xcol-ti])+1)).astype(int)
            norm_h = np.linalg.norm(h[:, (Lh+tau).astype(int)])
            """%norm_h = h(Lh+1)""" 
            indices = ((N+tau)%N+1).astype(int)
            tf0 = np.zeros((1, N), dtype=complex)
            tf1 = np.zeros((1, N), dtype=complex) 
            tf0[:,indices-1] = x[:, ti+tau-1]*np.conjugate(h[:, Lh+tau])/norm_h
            tf1[:,indices-1] = x[:, ti+tau-1]*np.conjugate(Dh[:, Lh+tau])/norm_h
            tf0 = np.fft.fft(tf0)
            #tf0 = tf0[:,0:int(N/2)]
            tf1 = np.fft.fft(tf1)
            #tf1 = tf1[:,0:int(N/2)]
            """% get the first order omega"""
            omega = np.zeros(tf1.shape)
            _, avoid_warn = np.nonzero(tf0)
            omega[:, avoid_warn]= np.round(np.imag(N*np.divide(tf1[:,avoid_warn],tf0[:,avoid_warn])/(2.0*math.pi)))
            sst = np.zeros((1,fLen),dtype=complex) 
            # int(N/2)+1: 501 
            for jcol in range(1,int(N/2)+1):
                if abs(tf0[0,jcol-1]) > Threshold:
                    jcolhat = int(jcol - omega[:,jcol-1])
                    if (jcolhat <= Hidx) & (jcolhat >= Lidx):  
                        sst[0,jcolhat-Lidx] = sst[0,jcolhat-Lidx]+tf0[0,jcol-1] 

            tfr[Idx, :] = tf0[0,0:int(N/2)]
            tfrsq[Idx,:] = sst
        return np.transpose(tfr), tfrtic, np.transpose(tfrsq), tfrsqtic     

    def ConceFT_sqSTFT_C(self, x, lowFreq, highFreq, alpha, hop, WinLen, dim, supp, MT, Second, Smooth):
        Hemi = 0
        h, Dh, _ = self.hermf(WinLen, dim, supp)
        tfr, tfrtic, tfrsq, tfrsqtic = self.sqSTFTbase(x, lowFreq, highFreq, alpha, hop, np.conj(np.array([h[0,:]])), np.conj(np.array([Dh[0,:]])), Smooth, Hemi)
        ConceFT = abs(tfrsq) 
        return tfr, tfrtic, tfrsq, ConceFT, tfrsqtic

    def SSTreconstruct_MP(self, THO, ABD): 
        data = [THO, ABD]
        q = multiprocessing.Queue()
        process = []
        
        # use multiprocess
        for i in range(len(data)):
              p = multiprocessing.Process(target=self.SSTreconstruct_process, args=(data[i], i, q))
              process.append(p)
              p.start()
        
        # get return value
        result = []
        for _ in range(len(process)):
            result.append(q.get())

        # wait process done
        for t in process:
              t.join()

        # Identify process ID
        if result[0][0] == 0:
            THO_SSTreconstruct = result[0][1]
            ABD_SSTreconstruct = result[1][1]
        else:
            THO_SSTreconstruct = result[1][1]
            ABD_SSTreconstruct = result[0][1]
        
        return THO_SSTreconstruct,ABD_SSTreconstruct

    def SSTreconstruct_process(self, x, patient_id, q): 
        tfr, tfrtic, tfrsq, ConceFT, tfrsqtic = self.ConceFT_sqSTFT_C(x - x.mean(), 0, 0.5, 0.001, 1, 123, 1, 6, 1, 0, 0)
        Sub_ConceFT = abs(ConceFT[25:200,:])
        curve = self.CurveExt_M_jit(Sub_ConceFT.T, 1)

        c = [n + 25 for n in curve]
        # c = curve + 25
        amABDTHO0 = [0]*len(x)
        # amABDTHO0 = np.zeros(len(x))

        for ii in range(len(amABDTHO0)):
            amABDTHO0[ii] = abs(sum(ConceFT[max(1,c[ii]-25)-1:c[ii]+25, ii]))

        package = [patient_id, amABDTHO0] 
        q.put(package)  

    def SSTreconstruct(self, THO, ABD): 

        stage_0 = time.time()

        ABDTHO=[ABD,THO] 
        amABDTHO =[0]*2
        for jj in range(2):
            x0 = ABDTHO[jj]
            x0mean = np.mean(x0)
            x = x0 - x0mean
            tfr, tfrtic, tfrsq, ConceFT, tfrsqtic = self.ConceFT_sqSTFT_C(x, 0, 0.5, 0.001, 1, 123, 1, 6, 1, 0, 0)
            Sub_ConceFT = abs(ConceFT[25:200,:])
            curve = self.CurveExt_M_jit(Sub_ConceFT.T, 1)
            #curve = CurveExt_M(Sub_ConceFT.T, 1)
            c = [n + 25 for n in curve]

            amABDTHO0 = [0]*len(x)

            for ii in range(len(amABDTHO0)):
                amABDTHO0[ii] = abs(sum(ConceFT[max(1,c[ii]-25)-1:c[ii]+25, ii]))

            # amABDTHO1 = smooth(amABDTHO0, 20)
            amABDTHO1 = RollingFucntion(amABDTHO0, 20, 'mean')
            amABDTHO1 = amABDTHO0

            amABDTHO[jj] = amABDTHO1

        amABD = amABDTHO[0]
        amTHO = amABDTHO[1]

        stage_1 = time.time()
        print("-----SSTreconstruct:"+str(stage_1 - stage_0)+"(s)-----")
        return amTHO, amABD

    def Feature_AR(self, STAGE, statePSG, THO, ABD, sp, ep):

        stage_0 = time.time()
        THO, ABD = self.SSTreconstruct_MP(THO, ABD)
        #THO, ABD = SSTreconstruct(THO, ABD)
        stage_1 = time.time()
        print("-----SSTreconstruct_MP:"+str(stage_1 - stage_0)+"(s)-----")

        AR_THO = np.full(len(statePSG), float("nan"))
        AR_ABD = np.full(len(statePSG), float("nan"))
        standWinidx = previousWinidx = currentWinidx = 0
        
        for ss in range(sp,ep+1):
            if (statePSG[ss-2] == self.NOR and statePSG[ss-1] != self.NOR) or ss == sp:
                standWinidx_s = int(self.fsDown*(ss-self.PreviousWindowARlen-1)/self.SLIDE+1)
                standWinidx_e = int(self.fsDown*(ss-1)/self.SLIDE)
                standWinidx = range(standWinidx_s-1,standWinidx_e)

            previousWinidx_s = int(self.fsDown*(ss-self.PreviousWindowARlen-1)/self.SLIDE+1)
            previousWinidx_e = int(self.fsDown*(ss-1)/self.SLIDE)
            previousWinidx = range(previousWinidx_s-1,previousWinidx_e)

            currentWinidx_s = int(self.fsDown*(ss-1)/self.SLIDE + 1)
            currentWinidx_e = int(self.fsDown*(ss+self.CurrentWindowARlen-1)/self.SLIDE)
            currentWinidx = range(currentWinidx_s-1,currentWinidx_e)
            index = int(np.fix(ss/self.SLIDE))-1
            stg = STAGE[index]
            AR_THO[ss-1], AR_ABD[ss-1] = self.One_stage_AR(THO, ABD, standWinidx, previousWinidx, currentWinidx, stg, statePSG[ss-2])
        return AR_THO, AR_ABD

    def Feature_FR(self, STAGE, statePSG, THO, ABD, sp, ep):

        FR_THO = [0]*len(statePSG)
        FR_ABD = [0]*len(statePSG)

        m = 128

        ErgN_s = int(np.fix(0.8/self.fsDown*m))-1
        ErgN_e = int(np.fix(1.5/self.fsDown*m))
        ErgD_s = int(np.fix(0.1/self.fsDown*m))-1
        ErgD_e = int(np.fix(0.8/self.fsDown*m))


        for ss in range(sp,ep+1):
            currentWinidx_s = int(self.fsDown*(ss-1)/self.SLIDE)
            currentWinidx_e = int(self.fsDown*(ss+self.CurrentWindowFRlen-1)/self.SLIDE)
            index = int(np.fix(ss/self.SLIDE))-1
            stg = STAGE[index]

            if stg == 11:
                FR_THO[ss-1] = float("nan")
                FR_ABD[ss-1] = float("nan")
            else:
                fqTHO = THO[currentWinidx_s:currentWinidx_e]
                fqABD = ABD[currentWinidx_s:currentWinidx_e]
                
                #smooth
                #fqTHO = fqTHO - smooth(fqTHO, 5*fsDown, 'loess')';
                #fqABD = fqABD - smooth(fqABD, 5*fsDown, 'loess')';
                fqTHO = fqTHO - self.smooth(fqTHO, 5*self.fsDown, 'mean')
                fqABD = fqABD - self.smooth(fqABD, 5*self.fsDown, 'mean')
                
                fTHO = np.square(abs(np.fft.fft(fqTHO, n=m)))
                fABD = np.square(abs(np.fft.fft(fqABD, n=m)))

                
                tmp_a_n = sum(fTHO[ErgN_s:ErgN_e])
                tmp_a_d = sum(fTHO[ErgD_s:ErgD_e])
                tmp_b_n = sum(fABD[ErgN_s:ErgN_e])
                tmp_b_d = sum(fABD[ErgD_s:ErgD_e])

                tmp_a = tmp_a_n/tmp_a_d
                tmp_b = tmp_b_n/tmp_b_d

                a = math.log10(tmp_a)
                b = math.log10(tmp_b)

                FR_THO[ss-1] = a
                FR_ABD[ss-1] = b

        for i in range(0,sp):
            FR_THO[i] = float("nan")
            FR_ABD[i] = float("nan")
        
        for j in range(ep,len(FR_THO)):
            FR_THO[j] = float("nan")
            FR_ABD[j] = float("nan")

        return FR_THO, FR_ABD

    def Feature_SpO2(self, STAGE, statePSG, SpO2, sp, ep):

        fsDown = 1    #SpO2 is recorded in 1Hz

        mu  = [0]*len(statePSG)
        drop = [0]*len(statePSG)
        Min = [0]*len(statePSG)
        Max = [0]*len(statePSG)

        for ss in range(sp,ep+1):

            currentWinidx_s = int(fsDown*np.fix((ss+self.SpO2delaylen-1)/self.SLIDE)) 
            currentWinidx_e = int(fsDown*np.fix((ss+self.SpO2delaylen+self.SpO2Windowlen-1)/self.SLIDE))
            index = int(np.fix(ss/self.SLIDE)) -1
            stg = STAGE[index]
            
            Firstorderdiff = np.diff(SpO2)
            
            if stg == 11:
                mu[ss -1 ]  = float("nan")
                drop[ss -1] = float("nan")
                Min[ss -1] = float("nan")
                Max[ss -1] = float("nan")
                
            else:
                
                mu[ss -1]  = np.median(SpO2[currentWinidx_s:currentWinidx_e])
                b = Firstorderdiff[currentWinidx_s-1:currentWinidx_e-1]
                drop[ss -1] =  np.sum(Firstorderdiff[currentWinidx_s-1:currentWinidx_e-1])
                
                Min[ss -1] =  np.min(SpO2[currentWinidx_s:currentWinidx_e])
                Max[ss -1] =  np.max(SpO2[currentWinidx_s:currentWinidx_e])
               
        for i in range(0,sp):
            mu[i] = float("nan")
            drop[i] = float("nan")
            Min[i] = float("nan")
            Max[i] = float("nan")
        
        for j in range(ep,len(mu)):
            mu[j] = float("nan")
            drop[j] = float("nan")
            Min[j] = float("nan")
            Max[j] = float("nan")

        return mu, drop, Min, Max

    def Feature_Cov(self, STAGE, statePSG, THO, ABD, sp, ep):

        Cov_THOABD = [0]*len(statePSG)

        for ss in range(sp,ep+1):
            currentWinidx_s = int(self.fsDown*(ss-1)/self.SLIDE)
            currentWinidx_e = int(self.fsDown*(ss+self.CurrentWindowFRlen-1)/self.SLIDE)
            index = int(np.fix(ss/self.SLIDE))-1
            stg = STAGE[index]

            if stg == 11:
                Cov_THOABD[ss -1] = float("nan")
            else:
                cwTHO = THO[currentWinidx_s:currentWinidx_e]
                cwABD = ABD[currentWinidx_s:currentWinidx_e]
                Cov_THOABD[ss -1] = np.dot(cwTHO,cwABD)/np.linalg.norm(cwTHO)/np.linalg.norm(cwABD)

        for i in range(0,sp):
            Cov_THOABD[i] = float("nan")
            
        for j in range(ep,len(Cov_THOABD)):
            Cov_THOABD[j] = float("nan")
        
        return Cov_THOABD

    def Feature_SDM(self, STAGE, statePSG, signal, sp, ep):

        SDM_T = [0]*len(statePSG)
        SDM_P = [0]*len(statePSG)

        peak, trough = PandT(signal.T, self.fsDown) #Extract the Peak and Trough from signal.

        for ss in range(sp,ep+1):
            
            if (statePSG[ss-2] == self.NOR and statePSG[ss-1] != self.NOR) or ss == sp:
                standWinidx_s = int(self.fsDown*(ss-self.PreviousWindowARlen-1)/self.SLIDE+1)
                standWinidx_e = int(self.fsDown*(ss-1)/self.SLIDE)
                standWinidx = range(standWinidx_s-1,standWinidx_e)

            previousWinidx_s = int(self.fsDown*(ss-self.PreviousWindowARlen-1)/self.SLIDE+1)
            previousWinidx_e = int(self.fsDown*(ss-1)/self.SLIDE)
            previousWinidx = range(previousWinidx_s-1,previousWinidx_e)

            currentWinidx_s = int(self.fsDown*(ss-1)/self.SLIDE + 1)
            currentWinidx_e = int(self.fsDown*(ss+self.CurrentWindowARlen-1)/self.SLIDE)
            currentWinidx = range(currentWinidx_s-1,currentWinidx_e)

            index = int(np.fix(ss/self.SLIDE))-1 #Similar structure with AR Feature.
            stg = STAGE[index] #double corfirm type

            SDM_T[ss], SDM_P[ss] = self.One_stage_SDM(signal, peak, trough, standWinidx, previousWinidx, currentWinidx, stg, statePSG[ss-2])

        for i in range(0,sp):
            SDM_T[i] = float("nan")
            SDM_P[i] = float("nan")
            
        for j in range(ep,len(SDM_T)):
            SDM_T[j] = float("nan")
            SDM_P[j] = float("nan")
        return SDM_T, SDM_P

    def One_stage_SDM(self, signal, peak, trough, standWinidx, previousWinidx, currentWinidx, stg, statePSG):
        stg = 13
        statePSG = 0
        
        trough_array = np.asarray(trough)
        peak_array = np.asarray(peak)
        signal_array = np.asarray(signal)

        if stg == 11:
            SDM_T = float("nan")
            SDM_P = float("nan")
        else:
            if statePSG == self.NOR:
                trough_array_tmp = trough_array >previousWinidx[0] 
                trough_array_index = trough_array_tmp < previousWinidx[-1]
                pT = trough_array[trough_array_index]
                trough_array_tmp = trough_array >currentWinidx[0] 
                trough_array_index = trough_array_tmp < currentWinidx[-1]
                cT = trough_array[trough_array_index]
                peak_array_tmp = peak_array >previousWinidx[0] 
                peak_array_index = peak_array_tmp < previousWinidx[-1]
                pP = peak_array[ peak_array_index ]
                peak_array_tmp = peak_array >currentWinidx[0] 
                peak_array_index = peak_array_tmp < currentWinidx[-1]
                cP = peak_array[ peak_array_index ]

                preT = signal_array[pT]
                preP =  signal_array[pP]
                curT = signal_array[cT]
                curP = signal_array[cP]
            
                if cT.size == 0 | pT.size == 0 :
                    SDM_T = -1
                else:
                    SDM_T = (np.median(curT) / np.median(preT))
            
                if  cP.size == 0 | pP.size == 0 :
                    SDM_P = -1
                else:
                    SDM_P = (np.median(curP) / np.median(preP))
            else:

                trough_array_tmp = trough_array >standWinidx[0] 
                trough_array_index = trough_array_tmp < standWinidx[-1]
                sT = trough_array[trough_array_index]
                trough_array_tmp = trough_array >currentWinidx[0] 
                trough_array_index = trough_array_tmp < currentWinidx[-1]
                cT = trough_array[trough_array_index]
                peak_array_tmp = peak_array >standWinidx[0] 
                peak_array_index = peak_array_tmp < standWinidx[-1]
                sP = peak_array[ peak_array_index ]
                peak_array_tmp = peak_array >currentWinidx[0] 
                peak_array_index = peak_array_tmp < currentWinidx[-1]
                cP = peak_array[ peak_array_index ]

                stdT = signal_array[sT]
                stdP =  signal_array[sP]
                curT = signal_array[cT]
                curP = signal_array[cP]
            
                if cT.size == 0 | sT.size == 0 :
                    SDM_T = -1
                else:
                    SDM_T = (np.median(curT) / np.median(stdT))
                
                if cP.size == 0 | sP.size == 0 :
                    SDM_P = -1
                else:
                    SDM_P = (np.median(curP) / np.median(stdP))

        return SDM_T, SDM_P  

    def FeatureExtraction(self, pnum, state, Channels):
        Features = [0]*pnum
        Features_Cov = [0]*pnum
        sp = self.PreviousWindowARlen + 1
        RightBoundary = max([self.CurrentWindowFRlen, self.CurrentWindowCovlen, self.CurrentWindowARlen])

        for i in range(pnum):

            ep = min(state[i][self.lens], self.SLIDE*len(Channels[i][1])/self.fsDown) - RightBoundary + 1
            THO = Channels[i][0]
            ABD = Channels[i][1]
            STAGE = Channels[i][2]
            SpO2 = Channels[i][4]
            PSG = state[i][self.psg]
            
            AR_THO, AR_ABD = self.Feature_AR(STAGE, PSG, THO, ABD, sp, ep)
            FR_THO, FR_ABD = self.Feature_FR(STAGE, PSG, THO, ABD, sp, ep)

            #SDM_T_THO, SDM_P_THO = Feature_SDM(STAGE, PSG, THO, sp, ep(i))
            #SDM_T_ABD, SDM_P_ABD = Feature_SDM(STAGE, PSG, ABD, sp, ep(i))
            SDM_T_THO, SDM_P_THO, SDM_T_ABD, SDM_P_ABD = 0, 0, 0, 0

            ep_SpO2 = ep + RightBoundary - self.SpO2delaylen - self.SpO2Windowlen + 1
            [mu_SpO2, drop_SpO2, min_SpO2, max_SpO2] = self.Feature_SpO2(STAGE, PSG, SpO2, sp, ep_SpO2)

            Features[i] = [AR_THO, AR_ABD, FR_THO, FR_ABD, SDM_T_THO, SDM_T_ABD, SDM_P_THO, SDM_P_ABD,mu_SpO2, drop_SpO2, min_SpO2, max_SpO2]
        
            Features_Cov[i] = self.Feature_Cov(STAGE, PSG, THO, ABD, sp, ep)

        return Features

    def CF_FeatureExtraction(self, state, Channels):
        sp = self.PreviousWindowARlen + 1
        RightBoundary = max([self.CurrentWindowFRlen, self.CurrentWindowCovlen, self.CurrentWindowARlen])
        ep = min(state[self.lens], self.SLIDE*len(Channels[1])/self.fsDown) - RightBoundary + 1
        ep = int(ep)
        THO = Channels[0]
        ABD = Channels[1]
        STAGE = Channels[2]
        SpO2 = Channels[4]
        PSG = state[self.psg]

        AR_THO, AR_ABD = self.Feature_AR(STAGE, PSG, THO, ABD, sp, ep)
        FR_THO, FR_ABD = self.Feature_FR(STAGE, PSG, THO, ABD, sp, ep)
        
        #SDM_T_THO, SDM_P_THO = Feature_SDM(STAGE, PSG, THO, sp, ep(i))
        #SDM_T_ABD, SDM_P_ABD = Feature_SDM(STAGE, PSG, ABD, sp, ep(i))
        SDM_T_THO, SDM_P_THO, SDM_T_ABD, SDM_P_ABD = 0, 0, 0, 0
        ep_SpO2 = ep + RightBoundary - self.SpO2delaylen - self.SpO2Windowlen + 1
        mu_SpO2, drop_SpO2, min_SpO2, max_SpO2 = self.Feature_SpO2(STAGE, PSG, SpO2, sp, ep_SpO2)
        return AR_THO, AR_ABD, FR_THO, FR_ABD, SDM_T_THO, SDM_T_ABD, SDM_P_THO, SDM_P_ABD,mu_SpO2, drop_SpO2, min_SpO2, max_SpO2
