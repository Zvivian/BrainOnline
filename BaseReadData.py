import os, hashlib, datetime, time
import threading
from typing import List, Optional, Union, Tuple
from abc import abstractmethod
from collections import deque
import socket, struct

import numpy as np
from numpy import ndarray

from .workers import ProcessWorker
from .logger import get_logger

logger_amp = get_logger('amplifier')
logger_marker = get_logger('marker')

class RingBuffer(deque):
    def __init__(self, size=1024):
        """Ring buffer object based on python deque data structure to store data.
        
        Parameters
        ----------
        size : int, optional
            maximum buffer size, by default 1024
        """
        super(RingBuffer, self).__init__(maxlen=size)
        self.max_size = size

    def isfull(self):
        """Whether current buffer is full or not.
        
        Returns
        -------
        boolean
        """
        return len(self) == self.max_size

    def get_all(self):
        """Access all current buffer value.
        
        Returns
        -------
        list
            the list of current buffer
        """
        return list(self)

class Marker(RingBuffer):
    def __init__(self, 
        interval: list, 
        srate: float, 
        events: Optional[Union[int, List[int]]] = None):
        if events is not None:
            self.events = [events] if isinstance(events, int) else events
            self.interval = [int(i*srate) for i in interval]
            self.latency = 0 if self.interval[1] <=0 else self.interval[1]
            max_tlim = max(0, self.interval[0], self.interval[1])
            min_tlim = min(0, self.interval[0], self.interval[1])
            size = max_tlim - min_tlim
            if min_tlim >= 0:
                self.epoch_ind = [self.interval[0], self.interval[1]]
            else:
                self.epoch_ind = [self.interval[0]-min_tlim, self.interval[1]-min_tlim]
        else:
            # continous mode
            self.interval = [int(i*srate) for i in interval]
            self.events = events
            self.latency = self.interval[1] - self.interval[0]
            size = self.latency
            self.epoch_ind = [0, size]

        self.countdowns = {}
        self.is_rising = True
        super().__init__(size=size)

    def __call__(self, event: int):
        # 使实例能够像函数一样被调用，同时不影响实例本身的生命周期
        # __call__()不影响一个实例的构造和析构，但是可以用来改变实例的内部成员的值。
        # add new countdown items
        if self.events is not None:
            event = int(event)
            if event != 0 and self.is_rising:
                if event in self.events:
                    # new_key = hashlib.md5(''.join(
                    #     [str(event), str(datetime.datetime.now())]).encode()).hexdigest()
                    new_key = ''.join([str(event), datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')])
                    self.countdowns[new_key] = self.latency + 1
                    logger_marker.info('find new event {}'.format(new_key))
                self.is_rising = False
            elif event == 0:
                self.is_rising = True
        else:
            if 'fixed' not in self.countdowns:
                self.countdowns['fixed'] = self.latency
        
        drop_items = []
        # update countdowns
        for key, value in self.countdowns.items():
            value = value - 1
            if value == 0:
                drop_items.append(key)
                logger_marker.info('trigger epoch for event {}'.format(key))
            self.countdowns[key] = value

        for key in drop_items:
            del self.countdowns[key]
        if drop_items and self.isfull():
            return True
        return False

    def get_epoch(self):
        data = super().get_all()
        return data[self.epoch_ind[0]:self.epoch_ind[1]]
    
class BaseAmplifier:
    """Base Ampifier class. 
    """
    def __init__(self):
        self._markers = {}
        self._workers = {}
        # event 用来实现多线程间同步通信
        # 如果Flag值为 False，当程序执行event.wait()方法时就会阻塞;
        # 如果Flag值为True时，程序执行event.wait()方法时不会阻塞继续执行。
        self._exit = threading.Event()

    @abstractmethod
    def recv(self) -> list:
        """the minimal recv data function, usually a package.
        """
        pass

    def start(self):
        """start the loop.
        """
        logger_amp.info('start the loop')
        self._t_loop = threading.Thread(
            target=self._inner_loop, 
            name='main_loop')
        self._t_loop.start()

    def _inner_loop(self):
        self._exit.clear()              # 将内部标志重置为假
        logger_amp.info('enter the inner loop')
        while not self._exit.is_set():  # 当且仅当内部标志为真时返回True 此时为False
            try:
                samples = self.recv()
                if samples:
                    self._detect_event(samples)
            except:
                pass
        logger_amp.info('exit the inner loop')

    def stop(self):
        """stop the loop.
        """
        logger_amp.info('stop the loop')
        self._exit.set()
        logger_amp.info('waiting the child thread exit')
        self._t_loop.join()
        self.clear()

    def _detect_event(self, samples):
        """More efficient way?"""
        for work_name in self._workers:
            logger_amp.info('process worker-{}'.format(work_name))
            marker = self._markers[work_name]
            worker = self._workers[work_name]
            for sample in samples:
                marker.append(sample)
                if marker(sample[-1]) and worker.is_alive():# 返回线程是否存活
                    worker.put(marker.get_epoch())
    
    def up_worker(self, name):
        logger_amp.info('up worker-{}'.format(name))
        self._workers[name].start()

    def down_worker(self, name):
        logger_amp.info('down worker-{}'.format(name))
        self._workers[name].stop()
        self._workers[name].clear_queue()

    def register_worker(self, 
            name: str, 
            worker: ProcessWorker, 
            marker: Marker):
        logger_amp.info('register worker-{}'.format(name))
        self._workers[name] = worker
        self._markers[name] = marker
    
    def unregister_worker(self, 
            name: str):
        logger_amp.info('unregister worker-{}'.format(name))
        del self._markers[name]
        del self._workers[name]

    def clear(self):
        logger_amp.info('clear all workers')
        worker_names = list(self._workers.keys())
        for name in worker_names:
            self.down_worker(name)
            self.unregister_worker(name)

'''  
def update_buffer
def reset_buffer    
def is_activated    
'''
       
class BaseReadData(BaseAmplifier):
    _COMMANDS = {
        'stop_connect': b'CTRL\x00\x01\x00\x02\x00\x00\x00\x00',
        'start_acq': b'CTRL\x00\x02\x00\x01\x00\x00\x00\x00',
        'stop_acq': b'CTRL\x00\x02\x00\x02\x00\x00\x00\x00',
        'start_trans': b'CTRL\x00\x03\x00\x03\x00\x00\x00\x00',
        'stop_trans': b'CTRL\x00\x03\x00\x04\x00\x00\x00\x00',
        'show_ver': b'CTRL\x00\x01\x00\x01\x00\x00\x00\x00',
        'show_edf': b'CTRL\x00\x03\x00\x01\x00\x00\x00\x00',
        'start_imp': b'CTRL\x00\x02\x00\x03\x00\x00\x00\x00',
        'req_version': b'CTRL\x00\x01\x00\x01\x00\x00\x00\x00',
        'correct_dc': b'CTRL\x00\x02\x00\x05\x00\x00\x00\x00',
        'change_setup': b'CTRL\x00\x02\x00\x04\x00\x00\x00\x00'
    }

    def __init__(self, 
            device_address: Tuple[str, int] = ('127.0.0.1', 4000), 
            srate: float = 1000, 
            num_chans: int =68):
        super().__init__()
        self.device_address = device_address            # IP,端口号
        self.srate = srate                             
        self.num_chans = num_chans                      # number of chans
        self.neuro_link = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # TCP link
        # the size of a package in neuroscan data is srate/25*(num_chans+1)*4 bytes
        self.pkg_size = srate/25*(num_chans+1)*4
        self.timeout = 2*25/self.srate

    def _unpack_header(self, b_header):
        ch_id = struct.unpack('>4s', b_header[:4])
        w_code = struct.unpack('>H', b_header[4:6])
        w_request = struct.unpack('>H', b_header[6:8])
        pkg_size = struct.unpack('>I', b_header[8:])
        return (ch_id[0].decode('utf-8'), w_code[0], w_request[0], pkg_size[0])

    def _unpack_data(self, num_chans, b_data):
        fmt = '>' + str((num_chans+1)*4) + 'B'
        samples = np.array(list(struct.iter_unpack(fmt, b_data)), dtype=np.uint8).view(np.int32).astype(np.float64)
        samples[:, -1] = samples[:, -1] - 65280
        samples[:, :-1] = samples[:, :-1]*0.0298*1e-6
        return samples.tolist()
   
    def _recv_fixed_len(self, num_bytes):
        fragments = []
        b_count = 0
        while b_count<num_bytes:
            try:
                chunk = self.neuro_link.recv(num_bytes - b_count)
            except socket.timeout as e:
                raise e
            b_count += len(chunk)
            fragments.append(chunk)

        b_data = b''.join(fragments)
        return b_data

    def send(self, message):
        self.neuro_link.sendall(message)

    def set_timeout(self, timeout):
        if self.neuro_link:
            self.neuro_link.settimeout(timeout)
        
    def connect_tcp(self):
        self.neuro_link.connect(self.device_address)
    
    def start_acq(self):
        self.send(self._COMMANDS['start_acq'])
        self.set_timeout(None)
        self.recv()
        self.recv()
        self.set_timeout(self.timeout)

    def recv(self):
        b_header = self._recv_fixed_len(12)
        header = self._unpack_header(b_header)
        samples = None
        if header[-1] != 0:
            b_data = self._recv_fixed_len(header[-1])
            samples = self._unpack_data(self.num_chans, b_data)
        return samples

    def stop_acq(self):
        self.set_timeout(None)
        self.send(self._COMMANDS['stop_acq'])
        self.recv()
        self.recv()
        self.set_timeout(self.timeout)

    def start_trans(self):
        self.send(self._COMMANDS['start_trans'])
        time.sleep(1e-2)
        self.start()

    def stop_trans(self):
        self.send(self._COMMANDS['stop_trans'])
        self.stop()

    def close_connection(self):
        self.send(self._COMMANDS['stop_connect'])
        if self.neuro_link:
            self.neuro_link.close()
            self.neuro_link = None