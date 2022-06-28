# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2021/4/02
# License: MIT License
"""
SSAVEP Feedback on NeuroScan.

"""
import os
from pyexpat import features
import time
from pathlib import Path
import joblib
import numpy as np

import mne
from mne.filter import resample
from mne.datasets.utils import _get_path

from scipy.special import softmax

from pylsl import StreamInfo, StreamOutlet

from BaseReadData import BaseReadData, Marker
from workers import ProcessWorker

from brainda.algorithms.decomposition.base import generate_filterbank
from brainda.algorithms.utils.model_selection import EnhancedLeaveOneGroupOut
from brainda.algorithms.decomposition.trca import EnsembleTRCA
from brainda.algorithms.decomposition.dsp import EnsembleDSP
from brainda.utils import upper_ch_names
from mne.io import read_raw_cnt

# from .dryscan import DryScan

def label_encoder(y, labels):
    new_y = y.copy()
    for i, label in enumerate(labels):
        ix = (y == label)
        new_y[ix] = i
    return new_y

def read_data(run_files, delay, chs, interval, labels):
    Xs, ys = [], []
    for run_file in run_files:
        raw = read_raw_cnt(run_file, preload=True, verbose=False)
        raw = upper_ch_names(raw)
        events = mne.events_from_annotations(raw, event_id=lambda x: int(x), verbose=False)[0]
        ch_picks = mne.pick_channels(raw.ch_names, chs, ordered=True)
        epochs = mne.Epochs(raw, events, event_id=labels, tmin=interval[0]+delay, tmax=interval[1]+delay, 
            baseline=None, picks=ch_picks, verbose=False)
        for label in labels:
            X = epochs[str(label)].get_data()[..., 1:]# 2000 points
            Xs.append(X)
            ys.append(np.ones((len(X)))*label)
    Xs = np.concatenate(Xs, axis=0)
    ys = np.concatenate(ys, axis=0)
    ys = label_encoder(ys, labels)
    
    return Xs, ys, ch_picks

def train_model(X, y, srate=1000):
    y = np.reshape(y, (-1))
    X = resample(X, up=256, down=srate)

    wp = [
        [30, 85], [60, 85]      # 刺激31~40Hz 二次谐波80
    ]
    ws = [
        [26, 90], [56, 90]
    ]

    filterweights = np.arange(1, 3)**(-1.25) + 0.25    
    filterbank = generate_filterbank(wp, ws, 256)
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)

    model =  EnsembleDSP(
            n_components=2, 
            filterbank=filterbank, 
            filterweights=filterweights)

    model = model.fit(X, y)

    return model

def model_predict(X, srate=1000, model=None):
    X = np.reshape(X, (-1, X.shape[-2], X.shape[-1]))
    X = resample(X, up=256, down=srate)
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)
    features = model.transform(X)
    return features  

def offline_validation(X, y, srate=1000):
    y = np.reshape(y, (-1))
    spliter = EnhancedLeaveOneGroupOut(return_validate=False)

    kfold_accs = []
    for train_ind, test_ind in spliter.split(X, y=y):
        X_train, y_train = np.copy(X[train_ind]), np.copy(y[train_ind])
        X_test, y_test = np.copy(X[test_ind]), np.copy(y[test_ind])
        
        model = train_model(X_train, y_train, srate=srate)
        features = model_predict(X_test, srate=srate, model=model)
        p_labels = np.argmax(features)
        kfold_accs.append(np.mean(p_labels==y_test))
    return np.mean(kfold_accs)

class FeedbackWorker(ProcessWorker):
    def __init__(self, run_files, delay, chs, 
            interval, labels, timeout, name):
        self.run_files = run_files
        self.delay = delay
        self.chs = chs
        self.interval = interval
        self.labels = labels
        self.timeout = timeout
        self.name = name
        super().__init__(timeout, name)
            
    def pre(self):
        X, y, ch_ind = read_data(self.run_files, self.delay, self.chs, self.interval, self.labels)
        print("Loding data from {}".format(filepath))
        acc = offline_validation(X, y, srate=1000)     # 计算离线准确率
        print("Current Model accuracy:{:.2f}".format(acc))
        self.estimator = train_model(X, y, srate=1000)
        self.ch_ind = ch_ind
        info = StreamInfo(
            name='feedback_ssavep', 
            type='Markers', 
            channel_count=1, 
            nominal_srate=0, 
            channel_format='int32', 
            source_id='ssavep_online_worker1')
        self.outlet = StreamOutlet(info)
        print('Waiting connection...')
        while not self._exit:
            if self.outlet.wait_for_consumers(1e-3):
                break
        print('Connected')

    def consume(self, data):
        data = np.array(data, dtype=np.float64).T
        data = data[self.ch_ind]
        p_labels = model_predict(data, srate=1000, model=self.estimator)
        p_labels = p_labels + 1
        p_labels = p_labels.tolist()
        print(p_labels)
        if self.outlet.have_consumers():
            self.outlet.push_sample(p_labels)
    
    def post(self):
        pass

if __name__ == '__main__':
    delay = 0.14
    srate = 1000
    interval = [0, 2]
    labels = list(range(1,11)) 
    cnts = 2
    filepath = "D:\\EEG\\datasets\\MNE-tunerl-data\\Wei2022"
    runs = list(range(1, cnts+1))                                   # .cnt个数
    run_files = ['{:s}\\{:d}.cnt'.format(filepath, run) for run in runs]
    chs = ['PZ', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'O1', 'OZ', 'O2']

    worker = FeedbackWorker(run_files, delay, chs, 
            interval, labels, timeout=5e-2, name='feedback_worker') # 处理程序
    marker = Marker([0.14, 2.14], srate, events=[1])                # 每种0.5s，共四种2s  打标签全为1
    
    ns = BaseReadData(
        device_address=('192.168.1.100', 4000),    
        srate=srate, 
        num_chans=68)                                               # NeuroScan parameter

    ns.connect_tcp()                                                # 与ns建立tcp连接
    ns.start_acq()                                                  # ns开始采集波形数据
    
    ns.register_worker('feedback_worker', worker, marker)           # register worker来实现在线处理
    ns.up_worker('feedback_worker')                                 # 开启在线处理进程
    time.sleep(0.5)                                                 # 等待 0.5s
    
    ns.start_trans()                                                # ns开始截取数据线程，并把数据传递数据给处理进程
    
    input('press any key to close\n')                               # 任意键关闭处理进程
    ns.down_worker('feedback_worker')                               # 关闭处理进程
    time.sleep(1)                                                   # 等待 1s

    ns.stop_trans()                                                 # ns停止在下截取线程
    ns.stop_acq()                                                   # ns停止采集波形数据
    ns.close_connection()                                           # 与ns断开连接
    ns.clear()
    print('bye')


    
    