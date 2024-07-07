import pickle
import copy
import time
from json import load
from multiprocessing import Condition
import os
from random import randint, random
import sys
from pathlib import Path
import platform
import pdb
from unittest import loader
from xml.dom import minicompat  # use pdb.set_trace() for debugging
import matplotlib.pyplot as plt
#plt.ion()
workPath = None
scriptPath = None
if os.getcwd()[-7:] == 'parable':
    workPath = str(Path(os.getcwd()).parent.parent.parent.joinpath("sample_python/work"))
    scriptPath = os.getcwd()
    os.chdir(workPath)  # mainlib은 gui사용시 반드시 work폴더에서 실행되야해서
else:
    # sample_python/work에서 실행되는 학습에 사용되는 다른 worker 프로세스들.
    workPath = os.getcwd()
    print(workPath)
    assert (workPath[-4:] == 'work')
    scriptPath = str(Path(workPath).parent.parent.joinpath("sample_SAMP/python/parable"))

sys.path.append(scriptPath)
sys.path.append(workPath)

import settings

settings.useConsole = True

import console.libmainlib as lua  # use make consolelib in the src folder.
import luamodule as lua  # see luamodule.py
import numpy as np
import torch

tl = lua.taesooLib()

from GAVAE.models import *
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, RandomSampler
from tensorboardX import SummaryWriter

vae_train = True


class TaesooLibDataset:
    def __init__(self):
        self.trackerData = lua.G_mat('tracker_data')  # matrixn
        self.validFrames = lua.G_ivec('validFrames').ref().copy()  # convert intvectorn to numpy array
        self.poseFeatureCache = lua.G_mat('poseFeatureCache')  # matrixn (reference)
        self.numBone = lua.M1_int('mLoader', 'numBone')
        self.rootTraj = lua.G_mat('rootTraj')
        self.PAST_WINDOW = lua.G_int('PAST_WINDOW')
        self.FUTURE_WINDOW = lua.G_int('FUTURE_WINDOW')
        self.DELAY = lua.G_int('DELAY')
        self.numTracker = (int)(self.trackerData.cols() / 7)

    def getInputPoseFeature(self, i):
        # referenceFrame=self.refCoord.row(iframe).toTransf() # use filterd rootTF

        tracker_data = self.trackerData

        def changeTrackerRotation(tracker_data_i, refCoord):
            changedTrackerData = tl.vectorn(self.numTracker * 9)
            for j in range(0, self.numTracker):
                tf = tracker_data_i.toTransf(7 * j)
                tf = refCoord.toLocal(tf)
                changedTrackerData.setTransf9(9 * j, tf)
            return changedTrackerData

        ifeature = tl.vectorn(self.numTracker * 9)
        # 변하는 reference frame사용!
        referenceFrame = tracker_data.row(i - 1).toTransf(0).project2D()  # use rootTF
        ifeature.assign(changeTrackerRotation(tracker_data.row(i), referenceFrame))
        assert (not ifeature.isnan())
        return ifeature

    def getOutputFeatureDim(self):
        numBone = self.numBone
        return (numBone - 1) * 6 + 3

    def getOutputPoseFeature(self, i):
        tracker_data = self.trackerData
        numBone = self.numBone
        ofeature = tl.vectorn(self.getOutputFeatureDim())

        referenceFrame = tracker_data.row(i - 1).toTransf(0).project2D()  # use rootTF
        # referenceFrame=self.refCoord.row(iframe).toTransf()
        ofeature.assign(self.poseFeatureCache.row(i))
        root = self.rootTraj.row(i).toTransf(0)
        root = referenceFrame.toLocal(root)
        ofeature.setTransf9(0, root)
        assert (not ofeature.isnan())
        return ofeature


def main():
    lua.init_console()
    lua.dostring('print("->  Luaenv initialized.")')
    lua.dofile(scriptPath + "/lua/extractFeatures.lua")
    lua.F('loadFeaturesToLuaGlobals')

    n2 = lua.M1_int('validFrames', 'size');

    dataset = TaesooLibDataset()

    latent_size = 300
    #latent_size2=543
    latent_size2=381
    #device = 'cpu'
    # device = 'cuda:0'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    condition_frame1 = 10
    assert (dataset.PAST_WINDOW - 1 >= 10)
    output_numframe1 = 10
    condition_frame2=5
    output_numframe2=10

    all_tracker_data = tl.matrixn(dataset.trackerData.rows(), dataset.numTracker * 9)
    all_tracker_data.setAllValue(0.0)
    all_mocap_data = tl.matrixn(dataset.trackerData.rows(), (dataset.numBone - 1) * 6 + 3)
    all_tracker_data.setAllValue(0.0)

    for i in range(1, all_tracker_data.rows()):
        all_tracker_data.row(i).assign(dataset.getInputPoseFeature(i))
        all_mocap_data.row(i).assign(dataset.getOutputPoseFeature(i))

    tracker_data = torch.from_numpy(all_tracker_data.ref()).float().to(device)
    if torch.isnan(tracker_data).any():
        pdb.set_trace()

    mocap_data = torch.from_numpy(all_mocap_data.ref()).float().to(device)

    frame_size = tracker_data.shape[0]
    tracker_size = tracker_data.shape[1]
    mocap_size = mocap_data.shape[1]
    print("mocap_size:", frame_size)
    print("pose_size:", mocap_size)
    print("tracker size:", tracker_size)
    mini_batch = 1024
    teacher_epochs = 100
    ramping_epochs = 100
    student_epochs = 100
    output_size = mocap_size
    iteration = 5
    vae_epochs = teacher_epochs + ramping_epochs + student_epochs
    print("Totally epoch:",vae_epochs)
    ae = TDGAVAE(tracker_size, condition_frame1,1024, latent_size, latent_size2,270, output_size,output_numframe1,output_size,output_numframe2 ).to(device)
    ae_optimizer = optim.Adam(ae.parameters(), lr=1e-4)
    scheduler_lr_ae = optim.lr_scheduler.StepLR(optimizer=ae_optimizer, step_size=20, gamma=0.95)

    all_indices = np.linspace(0, frame_size - 1, frame_size)

    good_mask = np.isin(all_indices, dataset.validFrames, assume_unique=True, invert=False)

    selectable_indices = all_indices[good_mask]

    writer=SummaryWriter()
    # print(selectable_indices)

    #epochs_list=[]
    #vae_loss_list=[]
    #plt.figure(figsize=(10,6))
    #plt.xlabel('Epochs')

    #plt.ylabel('VAE Loss')
    #plt.title('VAE Loss Over Epoch')
    #plt.grid(True)
    #line,=plt.plot([],[],label='VAE Loss')
    #plt.legend()
    #plt.show()
    if vae_train:
        ae.train()
        shape = (mini_batch, condition_frame1, tracker_size)
        history = torch.empty(shape).to(device)
        start = time.time()
        for ep in range(1, vae_epochs + 1):
            sampler = BatchSampler(SubsetRandomSampler(selectable_indices), mini_batch, drop_last=True)
            ep_recon_loss = 0
            for step, indices in enumerate(sampler):
                t_indices = torch.LongTensor(indices)
                condition_range = (t_indices.repeat((condition_frame1, 1)).t()
                                   + torch.arange(0, -1, -1).long()
                                   )
                output_range = condition_range.clone()
                # 현재 입출력 윈도우가 같은데 delay, future window가 고려되도록 수정 필요.
                for i in range(condition_frame1):
                    condition_range[:,
                    i] += i - condition_frame1  # first frame cannot be used because inputfeature uses frame i-1
                    output_range[:, i] += i - dataset.DELAY + 1
                    history[:, i, :].copy_(tracker_data[condition_range[:, i]])

                # t_indices += 4
                for offset in range(iteration):
                    ground_truth = mocap_data[output_range, :]
                    output,_,_ = ae(history[:,condition_frame1-1,:],history[:,condition_frame1-2,:],history[:,condition_frame1-3,:],history[:,condition_frame1-4,:],history[:,condition_frame1-5,:],history[:,condition_frame1-6,:],history[:,condition_frame1-7,:],history[:,condition_frame1-8,:],history[:,condition_frame1-9,:],history[:,condition_frame1-10,:])
                    #output,_,_=ae(history)
                    output = output.view(-1, output_numframe2, output_size)
                    recon_loss = F.mse_loss(output, ground_truth)
                    ae_optimizer.zero_grad()
                    (recon_loss).backward()
                    ae_optimizer.step()
                    ep_recon_loss += float(recon_loss) / iteration

            end = time.time()
            avg_vae_loss = ep_recon_loss / mini_batch
            scheduler_lr_ae.step()
            fps=int((ep/(end-start))*100)
            lr=ae_optimizer.param_groups[0]['lr']
            #epochs_list.append(ep)
            #vae_loss_list.append(avg_vae_loss)
            writer.add_scalar('Loss/VAE',avg_vae_loss,ep)
            writer.add_scalar('Learning Rate',lr,ep)
            writer.add_scalar('FPS',fps,ep)
            print("epochs : {ep} , vae_loss : {avg_vae_loss:0.08f} ,  lr : {lr:0.08f},  FPS : {FPS}".format(ep=ep,
                                                                                                            avg_vae_loss=avg_vae_loss,
                                                                                                            lr=
                                                                                                            ae_optimizer.param_groups[
                                                                                                                0][
                                                                                                                'lr'],
                                                                                                            FPS=int((
                                                                                                                                ep / (
                                                                                                                                    end - start)) * 100)))
            torch.save(copy.deepcopy(ae).cpu(), "TDGAVAE10.pt")
            #line.set_xdata(epochs_list)
            #line.set_ydata(vae_loss_list)
            #plt.draw()
            #plt.pause(0.01)


        #plt.ioff()
        #plt.savefig('vae_loss_over_epochs.png')
        #plt.show()
        writer.close()



if __name__ == "__main__":
    main()