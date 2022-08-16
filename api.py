"""This script is the test script for Deep3DFaceRecon_pytorch
"""
import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

import torch
import tensorflow as tf
import numpy as np
import cv2
import pyrender
import trimesh

# import os
import glob
import platform
import argparse
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
input_path = 'input'
reconstruct_path = 'output'
rasterize_path = 'rasterize'

if not os.path.exists(reconstruct_path):
	os.makedirs(reconstruct_path)

if not os.path.exists(rasterize_path):
	os.makedirs(rasterize_path)

def detect(img):

    img = np.array(img)
    color = (255, 255, 51) #bright yellow
    thickness = 3 

    #setup detector
    detector = MTCNN()

    #detect keypoints
    keypoints = detector.detect_faces(img)[0]['keypoints']

    #extract landmarks from keypoints
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

    #create numpy ndarray 
    landmark = np.array([[left_eye_x, left_eye_y], 
        [right_eye_x, right_eye_y], 
        [nose_x, nose_y], 
        [mouth_left_x, mouth_left_y], 
        [mouth_right_x, mouth_right_y]], 
        dtype='f')

    #draw landmarks and bounding box
#     bounding_box = detector.detect_faces(img)[0]['box']

#     cv2.rectangle(img,
#         (bounding_box[0], bounding_box[1]),
#         (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
#         color, thickness)

#     cv2.circle(img,(keypoints['left_eye']), thickness, color, -1)
#     cv2.circle(img,(keypoints['right_eye']), thickness, color, -1)
#     cv2.circle(img,(keypoints['nose']), thickness, color, -1)
#     cv2.circle(img,(keypoints['mouth_left']), thickness, color, -1)
#     cv2.circle(img,(keypoints['mouth_right']), thickness, color, -1)

#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # return landmark, img
    return landmark


def reconstruct(rank, opt, im, lm):
    device = torch.device(rank)
    torch.cuda.set_device(device)
    model = create_model(opt)
    model.setup(opt)
    model.device = device
    model.parallelize()
    model.eval()

    lm3d_std = load_lm3d(opt.bfm_folder) 
    
    
    W,H = im.size
    lm = lm.reshape([-1, 2])
    lm[:, -1] = H - 1 - lm[:, -1]
    _, im, lm, _ = align_img(im, lm, lm3d_std)
    
    im_tensor = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    lm_tensor = torch.tensor(lm).unsqueeze(0)
    
    data = {
        'imgs': im_tensor,
        'lms': lm_tensor
    }
    
    model.set_input(data)  # unpack data from data loader
    model.test()           # run inference
        
    recon_shape, tri, recon_tex = model.export_mesh()
        
    return recon_shape, tri, recon_tex

def rasterize(shape, index, texture, quality):
	
	if quality == 'low':
		size = 256
	elif quality == 'high':
		size = 1024
	else: 
		size = 512

	#set up pyrender scene
	scene = pyrender.Scene(bg_color = [0.0, 0.0, 0.0])

	#set up pyrender camera
	def camera_pose():
		centroid = [0.0, 0.0, 2.5]
		cp = np.eye(4)
		s2 = 1.0 / np.sqrt(2.0)

		cp[:3,:3] = np.array([
			[1.0, 0.0, 0.0],
			[0.0, 1.0, 0.0],
			[0.0, 0.0, 1.0]
		])

		hfov = np.pi / 3.0
		dist = scene.scale / (2.0 * np.tan(hfov))
		cp[:3,3] = dist * np.array([0.0, 0.0, 0.0]) + centroid

		return cp

	pc = pyrender.PerspectiveCamera(yfov=np.pi / 3.5, znear=0.5, zfar=50.0)
	oc = pyrender.OrthographicCamera(xmag = 1.0, ymag = 1.0, znear = 0.5, zfar = 50.0)
	npc = pyrender.Node(matrix=camera_pose(), camera=pc)
	noc = pyrender.Node(matrix=camera_pose(), camera=oc)

	scene.add_node(npc)
	scene.add_node(noc)
	#noc = orthographic, npc = perspective
	scene.main_camera_node = noc

	#set up pyrender light
	dlight = pyrender.DirectionalLight(color=[0.9, 0.75, 0.7], intensity=5.0)

	scene.add(dlight, pose=camera_pose())

	#load mesh
	input_mesh = trimesh.Trimesh(vertices = shape, faces = index, vertex_colors = texture, process = False)
	mesh = pyrender.Mesh.from_trimesh(input_mesh)
	nm = pyrender.Node(mesh=mesh, matrix=np.eye(4))
	scene.add_node(nm)
	
	#Offscreen Render
	r = pyrender.OffscreenRenderer(viewport_width=size, viewport_height=size, point_size=1.0)
	
	color, depth = r.render(scene)
	
	b,g,red = cv2.split(color)
	result = cv2.merge((red,g,b))
	
	scene.remove_node(nm)
	r.delete()

	#save result as PNG image
	# cv2.imwrite('pyrender.png', result)

	return result

def unmask(input_img):
    
    opt = TestOptions().parse()
    
    lm = detect(input_img)
    
    shape, tri, texture = reconstruct(0, opt, input_img, lm)
    
    recon_img = rasterize(shape, tri, texture, quality = 'low')
    
    return recon_img

if __name__ == '__main__':
    
    input_img = Image.open('input.png').convert('RGB')
    
    recon_img = unmask(input_img)
    
    cv2.imwrite('rasterized.png', recon_img)