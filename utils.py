import os
import time
import cv2
from PIL import Image


def get_time():
    secondsSinceEpoch = time.time()
    timeObj = time.localtime(secondsSinceEpoch)
    cur_time = str('%d-%d-%d %d:%d:%d' % (timeObj.tm_year, timeObj.tm_mon, timeObj.tm_mday, timeObj.tm_hour, timeObj.tm_min, timeObj.tm_sec))

    return cur_time

def mk_directory(usr_id):
    folder_path = "./usr_id_" + usr_id
    try:
        os.makedirs(folder_path + "/")
    except:
        pass

    return folder_path

def count_files(folder_path):
    list = os.listdir(folder_path)
    num_files = len(list)

    return num_files

def img_shrink(im):
    wid, hei = im.size
    wid_s = int(wid*0.5)
    hei_s = int(hei*0.5)
    #print(wid_s, hei_s)

    im_s = im.resize((wid_s, hei_s), Image.ANTIALIAS)
    
    return im_s

def contrast_up(im):
    # import matplotlib.pyplot as plt
    im2 = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
    im2[:, :, 0] = cv2.equalizeHist(im2[:, :, 0])

    return im2