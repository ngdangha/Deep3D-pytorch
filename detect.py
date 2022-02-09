import cv2
import os
import glob
from pathlib import Path
import shutil
import imutils

from mtcnn import MTCNN


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

train_val_path = ['datasets/vkist/training', 'datasets/vkist/validation']
no_landmark_move_path = 'no_landmark_found'

# save landmarks as textfile
def save_txt(path, lex, ley, rex, rey, nx, ny, mlx, mly, mrx, mry):
    with open(path, 'w') as file_path:
        file_path.write('%s\t%s\n' % (lex, ley))
        file_path.write('%s\t%s\n' % (rex, rey))
        file_path.write('%s\t%s\n' % (nx, ny))
        file_path.write('%s\t%s\n' % (mlx, mly))
        file_path.write('%s\t%s\n' % (mrx, mry))
    file_path.close()


def replaceImageNameToText(file_path):
    return file_path.split(os.path.sep)[-1].replace('.png', '.txt').replace('.PNG', '.txt').replace('.jpg',
                                                                                                    '.txt').replace(
        '.JPG', '.txt')


for individual_path in train_val_path:
    # set up input, output directory
    image_path = individual_path
    landmark_path = 'detections'

    # if not os.path.exists(landmark_path):
    #    os.makedirs(landmark_path)

    img_list = glob.glob(image_path + '/' + '*.png')
    img_list += glob.glob(image_path + '/' + '*.jpg')
    img_list += glob.glob(image_path + '/' + '*.JPG')

    # set up detector
    detector = MTCNN()
    # main process
    n = 0

    for file in img_list:
        landmark_save_name = str(replaceImageNameToText(file))
        available_detection_txt = Path(
            image_path, landmark_path, landmark_save_name)

        if available_detection_txt.is_file() and (not os.stat(available_detection_txt).st_size == 0):
            print(bcolors.WARNING + str(n) + " --- Landmark " +
                  str(landmark_save_name) + " existed. Skipping...")
            continue

        img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
        
        # face = detector.detect_faces(img)
        detect_result = detector.detect_faces(img)

        if not detect_result:
            print(bcolors.FAIL + str(n) + " --- [WARNING]: " + "No keypoint detected for: " + file)
            shutil.move(os.path.join(file), os.path.join(image_path, no_landmark_move_path, file.split(os.path.sep)[-1]))
            continue

        key_points = detect_result[0]['keypoints']
        
        left_eye_x = key_points["left_eye"][0]
        left_eye_y = key_points["left_eye"][1]
        
        right_eye_x = key_points["right_eye"][0]
        right_eye_y = key_points["right_eye"][1]
        
        nose_x = key_points["nose"][0]
        nose_y = key_points["nose"][1]
        
        mouth_left_x = key_points["mouth_left"][0]
        mouth_left_y = key_points["mouth_left"][1]
        
        mouth_right_x = key_points["mouth_right"][0]
        mouth_right_y = key_points["mouth_right"][1]
        
        save_txt(os.path.join(image_path, landmark_path, landmark_save_name),
                 left_eye_x, left_eye_y, right_eye_x, right_eye_y, nose_x, nose_y, mouth_left_x, mouth_left_y,
                 mouth_right_x, mouth_right_y)
        
        n += 1
        print(bcolors.OKGREEN + str(n) + " --- " + file + ": Ok")
