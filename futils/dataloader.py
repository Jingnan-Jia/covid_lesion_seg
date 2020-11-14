# -*- coding: utf-8 -*-
"""
3D Extension of the work in https://github.com/costapt/vess2ret/blob/master/util/data.py.
===========================================================================================
Created on Tue Apr  4 09:35:14 2017
@author: Jingnan
"""

import os
import random
import numpy as np
from jjnutils.image import Iterator
from jjnutils.vpatch import random_patch
import time
import glob
from functools import wraps
import queue
import threading
import jjnutils.util as cu

def rp_dcrt(fun):  # decorator to repeat a function until succeessful
    """
    A decorator to repeat this function until no errors raise.
    here, the function normally is a dataloader: iterator.next()
    sometimes, ct scans are empty or pretty smaller than patch size which can lead to errors.

    :param fun: a function which might meet errors
    :return: decorated function
    """

    @wraps(fun)
    def decorated(*args, **kwargs):
        next_fail = True
        while (next_fail):
            try:
                out = fun(*args, **kwargs)
                next_fail = False
            except:
                print('data load or patch failed, pass this data, load next data')
        return out

    return decorated


class QueueWithIndex(queue.Queue):
    def __init__(self, qsize, index_list, name):
        self.index_list = index_list
        self.name = name
        super(QueueWithIndex, self).__init__(qsize)


def get_img_extension(filename):
    extension = '.' + filename.split(".")[-1]
    if extension == ".gz":
        extension = ".nii.gz"
    return extension


def get_extension_and_filename(a_dir):
    files_list = cu.get_all_ct_names(a_dir)
    extension = get_img_extension(files_list[0])
    # Files inside a and b should have the same name. Images without a pair are discarded.
    files = set(x.split(extension)[0].split(a_dir + '/')[-1] for x in files_list)
    filenames = sorted(list(files))

    return extension, filenames


class ScanIterator(Iterator):
    """Class to iterate A and B 3D scans (mhd or nrrd) at the same time."""

    def __init__(self,
                 directory,
                 task=None,
                 a_dir_name='ori_ct', b_dir_name='gdth_ct',
                 sub_dir=None,
                 c_extension='.nrrd',
                 ptch_sz=None, ptch_z_sz=None,
                 tszzyx=None,
                 tspzyx=None,
                 data_argum=True,
                 patches_per_scan=5,
                 ds=2,
                 labels=None,
                 batch_size=1,
                 shuffle=True,
                 seed=None,
                 n=None,
                 recon_dir=None,
                 p_middle=None,
                 aux=None,
                 ptch_seed=None,
                 io=False,
                 pad=48):
        """
        Iterate through two directories at the same time.

        Files under the directory A and B with the same name will be returned
        at the same time.
        Parameters:
        - directory: base directory of the dataset. Should contain two
        directories with name a_dir_name and b_dir_name;
        - a_dir_name: name of directory under directory that contains the A
        images;
        - b_dir_name: name of directory under directory that contains the B
        images;
        - c_dir_name : this is the auxiliar output folder
        - a/b/c_extension : type of the scan: nrrd or mhd (no dicom available)

        - N: if -1 uses the entire dataset. Otherwise only uses a subset;
        - batch_size: the size of the batches to create;
        - shuffle: if True the order of the images in X will be shuffled;
        - ds: number of deep classifiers
        - labels: output classes

        """
        self.pad = pad
        self.data_type_list = [".mhd", ".mha", ".nrrd", ".nii", ".nii.gz"]
        self.io = io
        self.task = task

        if "2_in" in self.io or self.task == 'vessel':  # I hope to used p_middle for mtscale
            self.p_middle = p_middle
        else:
            self.p_middle = None

        self.sub_dir = sub_dir
        self.ds = ds
        self.labels = labels
        self.patches_per_scan = patches_per_scan
        self.tszzyx = tszzyx
        self.tspzyx = tspzyx
        self.ptch_sz = ptch_sz
        self.ptch_z_sz = ptch_z_sz
        self.ptch_seed = ptch_seed
        self.aux = aux
        self.running = True
        if self.aux:
            self.c_dir_name = 'aux_gdth'
            self.c_dir = os.path.join(directory, self.c_dir_name, sub_dir)
            self.c_extension = c_extension
        else:
            self.c_dir = None

        if self.task == 'recon':
            self.a_dir = os.path.join(directory, a_dir_name, recon_dir)
            # extension denotes ".mhd" instead of "mhd"; # filename denotes "example_ct
            self.a_extension, self.filenames = get_extension_and_filename(self.a_dir)

        else:
            self.a_dir = os.path.join(directory, a_dir_name, sub_dir)
            self.b_dir = os.path.join(directory, b_dir_name, sub_dir)
            self.a_extension, a_files = get_extension_and_filename(self.a_dir)
            self.b_extension, b_files = get_extension_and_filename(self.b_dir)
            self.filenames = sorted(list(set(a_files).intersection(set(b_files))))
        if 'train' in self.a_dir:
            self.state = 'train'
        else:
            self.state = 'monitor'
        if n:
            self.filenames = self.filenames[:n]

        print('task:', self.task)
        print('from this directory:', self.a_dir)
        print('these files are used ', self.filenames)
        if self.filenames is []:
            raise Exception('empty dataset')

        self.epoch_nb = 0

        self.data_argum = data_argum
        # self.shuffle = shuffle
        super(ScanIterator, self).__init__(len(self.filenames), batch_size, shuffle, seed)

    def _normal_normalize(self, scan):
        """returns normalized (0 mean 1 variance) scan"""
        scan = (scan - np.mean(scan)) / (np.std(scan))
        return scan

    def load_scan(self, file_name):
        """Load mhd or nrrd 3d scan thread safelly. Output is scan and spacing with shape (z,y,x)"""
        with self.lock:
            print(threading.current_thread().name + " get the lock, thread id: " + str(threading.get_ident()) +
                  " prepare to load data")
            scan, origin, spacing = cu.load_itk(file_name)  # all the output shape are (z, y, z)
            print(threading.current_thread().name + " load data successfully, release the lock")
        return np.expand_dims(scan, axis=-1), spacing  # size=(z,x,y,1)

    def _load_img_pair(self, idx):
        """Get a pair of images after padding and normalization if possible with spacing."""
        a_fname = self.filenames[idx] + self.a_extension
        print('start load file: ', a_fname)

        a, spacing = self.load_scan(file_name=os.path.join(self.a_dir, a_fname))  # (200, 512, 512, 1)
        a = np.pad(a, ((self.pad, self.pad), (self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode='constant',
                   constant_values=-3000)

        # a = np.array(a)
        a = cu.normalize(a)  # threshold to [-1000,400], then rescale to [0,1]
        a = self._normal_normalize(a)

        if self.task == 'recon':
            return a, a, spacing
        else:
            b_fname = self.filenames[idx] + self.b_extension
            b, _ = self.load_scan(file_name=os.path.join(self.b_dir, b_fname))  # (200, 512, 512, 1)
            b = np.pad(b, ((self.pad, self.pad), (self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode='constant',
                       constant_values=0)

            if not self.aux:
                return a, b, spacing
            else:
                c_fname = self.filenames[idx] + self.c_extension
                c, _ = self.load_scan(file_name=os.path.join(self.c_dir, c_fname))  # (200, 512, 512, 1)
                c = np.pad(c, ((self.pad, self.pad), (self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode='constant',
                           constant_values=0)
                return a, b, c, spacing

    @rp_dcrt
    def next(self):
        return None

    def get_ct_for_patching(self, idx):
        """
        a_hgh and b_hgh must exist. but:
        if trgt_sz and trgt_sp are None and not self.mtscale, a_low, b_low, c_low would be None
        if not self.aux, c_low, c_hgh would be None
        :param idx: index of ct scan
        :return: a_low, a_hgh, b_low, b_hgh, c_low, c_hgh
        """

        if self.aux:
            # with padding, shape of a,b,c_ori is (z,y,x,1), spacing (z,y,x)
            a_ori, b_ori, c_ori, spacing = self._load_img_pair(idx)
        else:
            # with padding, shape of a,b_ori is (z,y,x,1), spacing is (z,y,x)
            a_ori, b_ori, spacing = self._load_img_pair(idx)
        if self.task != 'recon':  # encode first, cu.downsample next. Otherwise we need to encode down and ori ct.
            b_ori = cu.one_hot_encode_3d(b_ori, self.labels)  # shape: (z,y,x,chn)
            c_ori = cu.one_hot_encode_3d(c_ori, [0, 1]) if self.aux else None  # shape: (z,y,x,2)

        if random.random() > 0.15:
            noise = np.random.normal(0, 0.05, a_ori.shape)
            a_ori += noise

        if self.data_argum:
            if self.aux:
                a_ori, b_ori, c_ori = cu.random_transform(a_ori, b_ori, c_ori)  # shape: (z,y,x,chn)
            else:
                a_ori, b_ori = cu.random_transform(a_ori, b_ori)  # shape: (z,y,x,chn)

        if self.io!="1_in_hgh_1_out_hgh" and not (any(self.tspzyx) or any(self.tszzyx)):
            raise Exception("io is: "+str(self.io)+" but did not set trgt_space_list or trgt_sz_list")

        a_low, a_hgh, b_low, b_hgh, c_low, c_hgh = None, None, None, None, None, None
        if "in_low" in self.io or "2_in" in self.io:
            a_low = cu.downsample(a_ori, is_mask=False,
                               ori_space=spacing, trgt_space=self.tspzyx,
                               ori_sz=a_ori.shape, trgt_sz=self.tszzyx,
                               order=1, labels=self.labels)  # shape: (z,y,x,chn)
        if "out_low" in self.io or "2_out" in self.io:
            b_low = cu.downsample(b_ori,is_mask=True,
                               ori_space=spacing, trgt_space=self.tspzyx,
                               ori_sz=b_ori.shape, trgt_sz=self.tszzyx,
                               order=0, labels=self.labels)  # shape: (z,y,x,chn)
            if self.aux:
                c_low = cu.downsample(c_ori,is_mask=True,
                                   ori_space=spacing, trgt_space=self.tspzyx,
                                   ori_sz=c_ori.shape, trgt_sz=self.tszzyx,
                                   order=0, labels=self.labels)  # shape: (z,y,x,2)

        if "in_hgh" in self.io or "2_in" in self.io:
            a_hgh = a_ori  # shape: (z,y,x,chn)
        if "out_hgh" in self.io or "2_out" in self.io:
            b_hgh = b_ori  # shape: (z,y,x,chn)
            if self.aux:
                c_hgh = c_ori

        return a_low, a_hgh, b_low, b_hgh, c_low, c_hgh  # shape (z,y,x,chn)

    def queue_data(self, q1, q2, i):
        with self.lock:
            print(threading.current_thread().name + " get the lock, thread id: " + str(threading.get_ident()) +
                  " prepare to get the index of data and load data latter")
            if self.state == "monitor" and not len(q1.index_list):
                print("task: " + self.task + " state: " + self.state + "worker_" + str(i) + " prepare put data but ")
                print( "index_list 1 is empty! the monitor thread has finished its job! stop this thread.")
                return True
            else:
                if not len(q1.index_list) and not len(q2.index_list):
                    pass  # all data have been sent, nothing to do now
                else:
                    if len(q1.index_list):
                        index = q1.index_list.pop()
                        current_q = q1
                    elif len(q2.index_list):
                        index = q2.index_list.pop()
                        current_q = q2
                    print("task: " + self.task + " state: " + self.state + "worker_" + str(i) +
                          " start to put data of " + str(index) + " into queue" + current_q.name + " for patching")
                    print("size of the queue: " + str(self.qmaxsize) + "occupied size: " + str(current_q.qsize()) +
                          "remaining index: " + str(current_q.index_list))
            print(threading.current_thread().name + "release the lock")

        ct_for_patching = self.get_ct_for_patching(index)

        with self.lock:
            print(threading.current_thread().name + " get the lock, thread id: " + str(threading.get_ident()) +
                  " prepare to put data to queue")
            t1 = time.time()
            current_q.put(ct_for_patching, timeout=600000)  # 160 hours  greater than the cost time of one epoch
            t2 = time.time()
            t = t2 - t1
            print("It cost this seconds to put the data into queue" + str(t))
            print("task: " + self.task + " state: " + self.state + "worker_" + str(
                i) + " successfully put data of " + str(index) +
                  " into queue" + current_q.name + " for patching")
            print("these index of data are waiting for loading: " + str(current_q.index_list))
            print("size of the queue: ", self.qmaxsize, "occupied size: ", current_q.qsize())
            print(threading.current_thread().name + "release the lock")

    def productor(self, i, q1, q2):
        # product ct scans for patching
        while True:
            if self.running:
                if self.epoch_nb % 2 == 0:  # even epoch number, q1 first
                    exit_flag = self.queue_data(q1, q2, i)
                else:
                    exit_flag = self.queue_data(q2, q1, i)
                if exit_flag:
                    return exit_flag
            else:
                print('running flag is set to false, this thread is stoped')
                return self.running

    def stop(self):
        self.running = False

    def join(self):
        for thd in self.thread_list:
            thd.join()

    def tune_axis(self, img):
        if type(img) is list:  # io is "2_out"
            img1, img2 = img[0], img[1]
            img1, img2 = np.rollaxis(img1, 0, 3), np.rollaxis(img2, 0, 3)
            img1, img2 = img1[np.newaxis, ...], img2 [np.newaxis, ...]
            return [img1, img2]
        else:
            img = np.rollaxis(img, 0, 3)  # (96, 144, 144, 1) -> (144, 144, 96, 1)
            img = img[np.newaxis, ...]  # (144, 144, 96, 1) -> (1, 144, 144, 96, 1)
            return img



    def generator(self, workers=5, qsize=5):
        index_sorted = list(range(self.n))
        if self.shuffle:
            index_list1 = random.sample(index_sorted, len(index_sorted))
            index_list2 = random.sample(index_sorted, len(index_sorted))
        else:
            index_list1 = list(range(self.n))
            index_list2 = list(range(self.n))
        self.qmaxsize = qsize
        q1 = QueueWithIndex(qsize, index_list1, name="q1")
        q2 = QueueWithIndex(qsize, index_list2, name="q2")
        self.thread_list = []
        for i in range(workers):
            thd = threading.Thread(target=self.productor, args=(i, q1, q2,))
            thd.name = self.task + str(i)
            thd.start()
            self.thread_list.append(thd)

            # self.thread_list.append(t)
        while True:
            # try:
            for _step in range(self.n):
                # try:
                q = q1 if self.epoch_nb % 2 == 0 else q2
                print("before q.get, qsize:", q.qsize(), q.index_list)
                time1 = time.time()

                print(threading.current_thread().name + " prepare to get data from queue")
                a_low, a_hgh, b_low, b_hgh, c_low, c_hgh  = q.get(timeout=6000)  # wait for several minitues for loading data
                # a/a2.shape (z,y,x,1) b/b2.shape (z,y,x,chn), c2.shape(z,y,x,2)
                time2 = time.time()
                print("it costs me this seconds to get the data: " + str(time2 - time1))
                print("after q.get, qsize:" + str(q.qsize()) + str(q.index_list))

                for _ in range(self.patches_per_scan):
                    if self.ptch_seed:
                        self.ptch_seed += 1
                    else:
                        self.ptch_seed = None

                    if self.ptch_sz is not None:
                        if self.aux:
                            # a/a2.shape (z,y,x,1) b/b2.shape (z,y,x,chn), c2.shape(z,y,x,2)
                            a_img, b_img, c_img = random_patch(a_low, a_hgh, b_low, b_hgh, c_low, c_hgh,
                                                               patch_shape=(self.ptch_z_sz, self.ptch_sz, self.ptch_sz),
                                                               p_middle=self.p_middle, task=self.task, io=self.io, ptch_seed=self.ptch_seed)
                        else:
                            a_img, b_img = random_patch(a_low, a_hgh, b_low, b_hgh, c_low, c_hgh,
                                                        patch_shape=(self.ptch_z_sz, self.ptch_sz, self.ptch_sz),
                                                        p_middle=self.p_middle, task=self.task, io=self.io, ptch_seed=self.ptch_seed)
                    else:
                        raise Exception("self.ptch_sz is None or self.ptch_sz == self.trgt_sz")

                    a_img = self.tune_axis(a_img)
                    b_img = self.tune_axis(b_img)
                    if self.aux:  # only for one output, b_img is an array
                        c_img = self.tune_axis(c_img)
                        if self.ds==2:
                            yield a_img, [b_img, c_img, b_img, b_img]
                        else:
                            yield a_img, [b_img, c_img]
                    else:
                        if self.ds == 2:  # only for one output, b_img is an array
                            yield a_img, [b_img, b_img, b_img]

                        else:
                            yield a_img, b_img

            self.epoch_nb += 1
            if self.shuffle:
                if not len(q1.index_list):
                    q1.index_list = random.sample(list(range(self.n)),
                                                  len(list(range(self.n))))  # regenerate the nex shuffle index
                elif not len(q2.index_list):
                    q2.index_list = random.sample(list(range(self.n)),
                                                  len(list(range(self.n))))  # regenerate the nex shuffle index
                # else:
                #     raise Exception("two queues are all not empty at the end of the epoch")
            else:
                if not len(q1.index_list):
                    q1.index_list = list(range(self.n))
                elif not len(q2.index_list):
                    q2.index_list = list(range(self.n))

            #             # import matplotlib.pyplot as plt
            #             # plt.figure()
            #             # plt.imshow(x1[0,72,:,:,0])
            #             # plt.savefig('x1_y.png')
            #             # plt.close()
            #             # plt.figure()
            #             # plt.imshow(x2[0,72, :, :, 0])
            #             # plt.savefig('x2_y.png')
            #             # plt.close()

