#!/usr/bin/env python
# ---------------------------- multiview to views ---------------------------- #
# This tool takes a multi-view image (a previously generated lenticular image) #
# and applies the reverse process, that is, it recovers and reconstructs the   #
# set of separate views that generated the multi-view image.                   #
#                                                                              #
# The tool is executed from the terminal and the input parameters are passed   #
# in the same command line                                                     #
#                                                                              #
# Args:                                                                        #
#   - <visual.json>                                                            #
#     Json file where the visual parameters that were used to generate the     #
#     multi-view image are established, such as: slope of the lenticulas and   #
#     DPI.                                                                     #
#   - <path-to-multiview-image>                                                #
#     Path to the input multivist image. Include the name of the multi-view    #
#     image.                                                                   #
#   - <number-of-views>                                                        #
#     Number of views to recover.                                              #
#   - <output-folder>                                                          #
#     Output folder where the recovered and reconstructed views are saved.     #
#                                                                              #
# Notes:                                                                       #
# The formulas derive from the concept of using a linear quilt, where all view #
# images are placed side-by-side, i.e. with only 1 tile vertically. This       #
# simplifies the formulas quite a bit and is conceptually slightly easier as a #
# 1D tile index becomes the same as the view index.                            #
#                                                                              #
# Partly based on https:                                                       #
# //github.com/lonetech/LookingGlass/blob/master/quiltshader.glsl (for which   #
# no license seems to be specified).                                           #
# -----------------------------------------------------------------------------#


import os, sys, shutil, json
from math import floor, cos, atan
from PIL import Image
from PIL import ImageFilter


# --- How to use display

def usage():
    print('usage: %s <visual.json> <path-to-multiview-image> <number-of-views> <output-folder> <only-recovery>' % sys.argv[0])
    print()
    sys.exit(-1)
    
if len(sys.argv) != 6:
    usage()


#--- initializations

json_name = sys.argv[1]
image_name = sys.argv[2]
n_views = int(sys.argv[3])
output_folder = sys.argv[4]
only_recovery = bool(sys.argv[5])


# --- configurations

# the resize factor
resize_factor = .2

# load visual.json file and get the visual parameters
config = json.loads(open(json_name, 'rt').read())
DPI = int(config['DPI']['value'])
slope = config['slope']['value']
pitch = n_views
# pitch = config['pitch']['value']

# open and load the input multi-view image
multiview_image = Image.open(image_name)
multiview_pixel = multiview_image.load()
multiview_width, multiview_height = multiview_image.size 

# calibration
screenInches = multiview_width / DPI    # physical image width
pitch = pitch * screenInches * cos(atan(1.0/slope))
tilt = multiview_height/(multiview_width * slope)
subpixel = 1.0 / (3*multiview_width) * pitch


#--- create the view images

print('creating the view images ...')

view_images = []    # For holding references to image objects
view_pixels = []    # For PixelAccess objects

view_width = int(multiview_width * resize_factor)
view_height = int(multiview_height * resize_factor)

for _ in range(n_views):
    view_image = Image.new('RGB', (view_width, view_height))
    view_images.append(view_image)
    view_pixels.append(view_image.load())


# --- create the result folder

if os.path.isdir(output_folder):
  shutil.rmtree(output_folder)
os.mkdir(output_folder)


# --- define some necessary functions

def determine_view(a):
    res = n_views - 1
    a = a%1 * n_views
    res -= floor(a)
    return res
    
def pixel_color(u, v, r, g, b):
    
    # XXX simplified to use the same i value for each subpixel. Seems to
    # work, as the subpixels can still get different views, but not sure
    # this is fully equivalent to what the Lenticular shader does :)
    i = int(u * view_width)
    j = int(v * view_height)
    
    a = (u + (1.0 - v) * tilt) * pitch
    
    # Red
    view = determine_view(a)
    img = view_pixels[view]
    img[i,j] = (r, g, b)
    
    # Green
    view = determine_view(a+subpixel)
    img = view_pixels[view]
    img[i,j] = (r, g, b)
    
    # Blue
    view = determine_view(a+2*subpixel)
    img = view_pixels[view]
    img[i,j] = (r, g, b)
    

# --- view image recovery: pixel by pixel

print('view image recovery: pixel by pixel ...')

for j in range(multiview_height):
    v = (j) / multiview_height
    
    for i in range(multiview_width):
        u = (i) / multiview_width
        
        r,g,b = multiview_pixel[i,j]
        if not (r==0 or g==0 or b==0):
            pixel_color(u, v, r, g, b)


# --- full reconstruction and save the view images

print('full reconstruction and save the view images ...')

for view in range(n_views):

    # get the path to save view image file
    view_file = 'view_image_%d.bmp' % ( view + 1)
    view_path = '%s/%s' %(output_folder, view_file)

    # full reconstruction of the view image
    view_image = view_images[view]
    if not only_recovery:
        view_image = view_image.filter(ImageFilter.MedianFilter(5));
        # view_image = view_image.filter(ImageFilter.MaxFilter(3));
        # view_image = view_image.filter(ImageFilter.DETAIL)
        # view_image = view_image.filter(ImageFilter.EDGE_ENHANCE)
        # view_image = view_image.filter(ImageFilter.EDGE_ENHANCE_MORE)
    
    # resize the view image to multi-view size
    view_image = view_image.resize(multiview_image.size, Image.ANTIALIAS)

    # saving the view image
    view_image.save(view_path)
    print( view_path, 'saved' )

#print('Tile size %d x %d' % (FRAME_WIDTH, FRAME_HEIGHT))