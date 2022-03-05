"""This script is the test script for Deep3DFaceRecon_pytorch
"""
import torch
import tensorflow as tf
import os
import numpy as np
import cv2
import time

from mtcnn import MTCNN
from util.load_mats import load_lm3d
from data.flist_dataset import default_flist_reader
from scipy.io import loadmat, savemat
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import MyVisualizer
from util.preprocess import align_img
from PIL import Image
from util.util import save_landmark

#set up path
image_test_path = 'images'
root = 'input'
reconstruct_path = 'output'
rasterize_path = 'rasterize'

if not os.path.exists(reconstruct_path):
	os.makedirs(reconstruct_path)

if not os.path.exists(rasterize_path):
	os.makedirs(rasterize_path)

def get_data_path(path):

    im_path = [os.path.join(path, i) for i in sorted(os.listdir(path)) if i.endswith('png') or i.endswith('jpg') or i.endswith('PNG') or i.endswith('JPG')]
    lm_path = [i.replace('png', 'txt').replace('jpg', 'txt').replace('PNG', 'txt').replace('JPG', 'txt') for i in im_path]
    # lm_path = [os.path.join(i.replace(i.split(os.path.sep)[-1],''),'detections',i.split(os.path.sep)[-1]) for i in lm_path]

    return im_path, lm_path

def read_data(im_path, lm_path, lm3d_std, to_tensor=True):
    # to RGB 
    im = Image.open(im_path).convert('RGB')
    W,H = im.size
    lm = np.loadtxt(lm_path).astype(np.float32)
    lm = lm.reshape([-1, 2])
    lm[:, -1] = H - 1 - lm[:, -1]
    _, im, lm, _ = align_img(im, lm, lm3d_std)

    if to_tensor:
        im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        lm = torch.tensor(lm).unsqueeze(0)
    
    return im, lm

def detect(path):
    detector = MTCNN()
    count = 0
    landmark_tic = time.perf_counter()
    im_path = [os.path.join(path, i) for i in sorted(os.listdir(path)) if i.endswith('png') or i.endswith('jpg') or i.endswith('PNG') or i.endswith('JPG')]

    for i in range(len(im_path)):
        count += 1
        print(count)
        img = cv2.cvtColor(cv2.imread(im_path[i]), cv2.COLOR_BGR2RGB)
        keypoints = detector.detect_faces(img)[0]['keypoints']

        left_eye_x = keypoints["left_eye"][0]
        left_eye_y = keypoints["left_eye"][1]

        right_eye_x = keypoints["right_eye"][0]
        right_eye_y = keypoints["right_eye"][1]

        nose_x = keypoints["nose"][0]
        nose_y = keypoints["nose"][1]

        mouth_left_x = keypoints["mouth_left"][0]
        mouth_left_y = keypoints["mouth_left"][1]

        mouth_right_x = keypoints["mouth_right"][0]
        mouth_right_y = keypoints["mouth_right"][1]

        #save detected landmark as text file
        save_landmark(os.path.join(path, im_path[i].split(os.path.sep)[-1].replace('.png','.txt').replace('.PNG','.txt').replace('.jpg','.txt').replace('.JPG','.txt')), 
            left_eye_x, left_eye_y, 
            right_eye_x, right_eye_y, 
            nose_x, nose_y, 
            mouth_left_x, mouth_left_y, 
            mouth_right_x, mouth_right_y) 


        landmark = np.array([[left_eye_x, left_eye_y], 
            [right_eye_x, right_eye_y], 
            [nose_x, nose_y], 
            [mouth_left_x, mouth_left_y], 
            [mouth_right_x, mouth_right_y]], 
            dtype='f')
            
    landmark_toc = time.perf_counter()
    landmark_time = landmark_toc - landmark_tic

    return landmark, landmark_time


def main(rank, opt, name):
    device = torch.device(rank)
    torch.cuda.set_device(device)
    model = create_model(opt)
    model.setup(opt)
    model.device = device
    model.parallelize()
    model.eval()
    visualizer = MyVisualizer(opt)

    im_path, lm_path = get_data_path(name)
    lm3d_std = load_lm3d(opt.bfm_folder) 

    recon_tic = time.perf_counter()

    for i in range(len(im_path)):
        print(i, im_path[i])
        img_name = im_path[i].split(os.path.sep)[-1].replace('.png','').replace('.jpg','').replace('.PNG','').replace('.JPG','')
        if not os.path.isfile(lm_path[i]):
            continue
        im_tensor, lm_tensor = read_data(im_path[i], lm_path[i], lm3d_std)
        data = {
            'imgs': im_tensor,
            'lms': lm_tensor
        }

        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        visualizer.display_current_results(rasterize_path, visuals, 0, opt.epoch, dataset=name.split(os.path.sep)[-1], 
            save_results=True, count=i, name=img_name, add_image=False)

        # model.save_mesh(os.path.join(visualizer.img_dir, name.split(os.path.sep)[-1], 'epoch_%s_%06d'%(opt.epoch, 0),img_name+'.obj')) # save reconstruction meshes
        # model.save_coeff(os.path.join(visualizer.img_dir, name.split(os.path.sep)[-1], 'epoch_%s_%06d'%(opt.epoch, 0),img_name+'.mat')) # save predicted coefficients

        # model.save_mesh(os.path.join(reconstruct_path, img_name+'.obj')) # save reconstruction meshes
        # model.save_obj(os.path.join(reconstruct_path, img_name+'.obj'))
        # model.save_shape_txt(os.path.join(reconstruct_path, img_name+'.txt'))

    recon_toc = time.perf_counter()
    recon_time = recon_toc - recon_tic

    return recon_time

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options

    # lm, lm_time = detect(root)
    # print(f"Detected Landmarks in {lm_time:0.4f} seconds")

    recon_time = main(0, opt, opt.img_folder)
    print(f"Created meshes in {recon_time:0.4f} seconds")
