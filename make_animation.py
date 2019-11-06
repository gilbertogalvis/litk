#!/usr/bin/env python

import cv2
import os, sys

def usage():
    print('usage: %s <view-folder> <video-name>' % sys.argv[0])
    print()
    sys.exit(-1)
    
if len(sys.argv) != 3:
    usage()

view_folder = sys.argv[1]
video_name = sys.argv[2]

nimages = len(os.listdir(view_folder))
images = ['view_image_%d.bmp' %(i+1) for i in range(nimages)]
images *= 2

frame = cv2.imread(os.path.join(view_folder, images[0]))
height, width, layers = frame.shape

fourcc  = cv2.VideoWriter_fourcc('M','J','P','G')
video = cv2.VideoWriter(video_name, fourcc, 15, (width,height), True)

for i, image in enumerate(images):
    print('frame: %d' %i)
    video.write(cv2.imread(os.path.join(view_folder, image)))

cv2.destroyAllWindows()
video.release()