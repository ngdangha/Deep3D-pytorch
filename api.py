"""This script is the test script for Deep3DFaceRecon_pytorch
"""

import torch
import tensorflow as tf
import numpy as np
import cv2
import trimesh

import os
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

from flask import Flask, request, jsonify
import base64
import face_alignment

app = Flask(__name__)

#set up path
input_path = 'input'
reconstruct_path = 'output'
rasterize_path = 'rasterize'

if not os.path.exists(reconstruct_path):
	os.makedirs(reconstruct_path)

if not os.path.exists(rasterize_path):
	os.makedirs(rasterize_path)

    
#set up model
opt = TestOptions().parse()    
device = torch.device(0)
torch.cuda.set_device(device)
model = create_model(opt)
model.setup(opt)
model.device = device
model.parallelize()
model.eval()
lm3d_std = load_lm3d('BFM') 
visualizer = MyVisualizer(opt)
    
#setup detector
# detector = MTCNN()
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cpu', face_detector='blazeface')

#detect landmark function
def detect(img):
#     global detector_mtcnn
    
#     detector_mtcnn = MTCNN()
    
    image = np.array(img)
    
    # print(detector)

    #detect keypoints
    preds = fa.get_landmarks_from_image(image)[0]

    #extract landmarks from keypoints
    left_eye_x = (preds[37][0] + preds[40][0])/2
    left_eye_y = (preds[37][1] + preds[40][1])/2

    right_eye_x = (preds[43][0] + preds[46][0])/2
    right_eye_y = (preds[43][1] + preds[46][1])/2

    nose_x = (preds[30][0] + preds[33][0])/2
    nose_y = (preds[30][1] + preds[33][1])/2

    mouth_left_x = preds[48][0]
    mouth_left_y = preds[48][1]

    mouth_right_x = preds[54][0]
    mouth_right_y = preds[54][1]

    #create numpy ndarray 
    landmark = np.array([[left_eye_x, left_eye_y], 
        [right_eye_x, right_eye_y], 
        [nose_x, nose_y], 
        [mouth_left_x, mouth_left_y], 
        [mouth_right_x, mouth_right_y]], 
        dtype='f')

    # return landmark, img
    return landmark


#reconstruct 3d face function
def reconstruct(im, lm):
    
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


#rasterize 3d face
def rasterize():
    visuals = model.get_current_visuals()  # get image results
    result = visualizer.save_img(visuals)

    b,g,red = cv2.split(result)
    result = cv2.merge((red,g,b))

    return result


#main api function
def unmask(input_img):
    
    lm = detect(input_img)
    
    shape, tri, texture = reconstruct(input_img, lm)
    
    recon_img = rasterize()
    
    return recon_img


def loadBase64Img(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def load_image(img):
	exact_image = False; base64_img = False; url_img = False

	if type(img).__module__ == np.__name__:
		exact_image = True

	elif len(img) > 11 and img[0:11] == "data:image/":
		base64_img = True

	elif len(img) > 11 and img.startswith("http"):
		url_img = True

	#---------------------------

	if base64_img == True:
		img = loadBase64Img(img)

	elif url_img:
		img = np.array(Image.open(requests.get(img, stream=True).raw))

	elif exact_image != True: #image path passed as input
		if os.path.isfile(img) != True:
			raise ValueError("Confirm that ",img," exists")

		img = cv2.imread(img)

	return img

#main

@app.route("/3dface", methods=["POST"])
def generate():
    req = request.get_json()
    
    img_input = ""
    if "img" in list(req.keys()):
        img_input = req["img"]

    validate_img = False
    if len(img_input) > 11 and img_input[0:11] == "data:image/":
        validate_img = True

    if validate_img != True:
        return jsonify({"result": {'message': 'Vui lòng truyền ảnh dưới dạng Base64'}}), 400
    
    input_img = load_image(img_input)
    
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    
    input_img = Image.fromarray(input_img)
    
    recon_img = unmask(input_img)
    
    retval, buffer = cv2.imencode('.jpg', recon_img)
    jpg_as_text = base64.b64encode(buffer)
    
    return jsonify({ "result": "data:image/jpeg;base64," + str(jpg_as_text)[2:-1] }), 200


app.run(host="0.0.0.0", port=5000)