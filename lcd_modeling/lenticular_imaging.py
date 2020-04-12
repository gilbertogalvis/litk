import os, sys
import pandas as pd
import numpy as np
import cv2 as cv
from PIL import Image, ImageDraw
from math import pi, tan
from astropy.convolution import Gaussian2DKernel
# Image.MAX_IMAGE_PIXELS = 1167220000


# - reading csv functions

convert1 = lambda x: tuple(int(ix) for ix in x.replace('(', '').replace(')', '').split(' '))
convert2 = lambda x: tuple(convert1(ix) for ix in x.replace('(', '').replace(')', '').split(';'))

apply_res_ = lambda x, res: tuple(ix//res for ix in x)
apply_res = lambda x, res: tuple(apply_res_(ix, res) for ix in x)

def get_params(filename):
    converters = {
        'pixel size':convert1,
        'red xy':convert2,
        'green xy':convert2,
        'blue xy':convert2,
        'red size':convert2,
        'green size':convert2,
        'blue size':convert2,
        'border':convert1,
        }
    data = pd.read_csv(filename, converters=converters)
    data.head(4)

    intname = list(data['input name'])
    outname = list(data['output name'])
    res = list(data['resolution'])
    pixsize = list(data['pixel size'])
    pixshift = list(data['pixel shift'])
    redxy = list(data['red xy'])
    greenxy = list(data['green xy'])
    bluexy = list(data['blue xy'])
    redsize = list(data['red size'])
    greensize = list(data['green size'])
    bluesize = list(data['blue size'])
    border = list(data['border'])

    i = 0
    params = list()
    for rxy, gxy, bxy in zip(redxy, greenxy, bluexy):

        rxylen, gxylen, bxylen = len(rxy), len(gxy), len(bxy)
        maxlen = max(rxylen, gxylen, bxylen)

        if len(pixsize[i]) == 1:
            pixsize[i] = tuple((pixsize[i][0], pixsize[i][0]))

        reds = redsize[i]
        if (len(reds) == 1) and maxlen > 1:
            redsize[i] = list(reds[0] for _ in range(maxlen))

        greens = greensize[i]
        if (len(greens) == 1) and maxlen > 1:
            greensize[i] = list(greens[0] for _ in range(maxlen))

        blues = bluesize[i]
        if (len(blues) == 1) and maxlen > 1:
            bluesize[i] = list(blues[0] for _ in range(maxlen))

        gxy = greenxy[i]
        if (len(gxy) == 1) and maxlen > 1:
            greenxy[i] = list(gxy[0] for _ in range(maxlen))

        rxy = redxy[i]
        if (len(rxy) == 1) and maxlen > 1:
            redxy[i] = list(rxy[0] for _ in range(maxlen))

        bxy = bluexy[i]
        if (len(bxy) == 1) and maxlen > 1:
            bluexy[i] = list(bxy[0] for _ in range(maxlen))

        params.append({
            'intname': intname[i],
            'outname': outname[i],
            'res': res[i],
            'pixsize': apply_res_(pixsize[i], res[i]), # pixsize[i],
            'pixshift': pixshift[i],
            'redxy': apply_res(redxy[i], res[i]),
            'greenxy': apply_res(greenxy[i], res[i]),
            'bluexy': apply_res(bluexy[i], res[i]),
            'redsize': apply_res(redsize[i], res[i]),
            'greensize': apply_res(greensize[i], res[i]),
            'bluesize': apply_res(bluesize[i], res[i]),
            'border': apply_res_(border[i], res[i]),
        })
        i += 1
    return params


# - pattern generation function

def get_pattern(params, rgb):

    # - inputs

    pixsize = params['pixsize']
    redxy = params['redxy']
    greenxy = params['greenxy']
    bluexy = params['bluexy']
    redsize = params['redsize']
    greensize = params['greensize']
    bluesize = params['bluesize']
    border = params['border']

    pattern = list()

    for rxy, gxy, bxy, rs, gs, bs in zip(redxy, greenxy, bluexy, redsize, greensize, bluesize):

        patt = Image.new('RGB', pixsize)
        draw = ImageDraw.Draw(patt)

        # red subpixel
        rend = rxy[0] + rs[0], rxy[1] + rs[1]
        draw.rectangle((rxy, rend), fill=(rgb[0],0,0))

        # green subpixel
        gend = gxy[0] + gs[0], gxy[1] + gs[1]
        draw.rectangle((gxy, gend), fill=(0,rgb[1],0))

        # blue subpixel
        bend = bxy[0] + bs[0], bxy[1] + bs[1]
        draw.rectangle((bxy, bend), fill=(0,0,rgb[2]))

        # add borders
        patt = add_borders(patt, border)

        pattern.append(patt)

    return pattern


# - image generation function

def load_image(params):

    input_image = Image.open('input/%s' %params['intname'])
    input_image_ = input_image.load()

    pixsize = params['pixsize']
    outsize = input_image.size
    pixshift = params['pixshift']

    fullsize = outsize[0] * pixsize[0], outsize[1] * pixsize[1]
    fullimg = Image.new('RGB', fullsize)
    shift = np.int0( pixsize[0] / pixshift )

    for h in range(outsize[1]):
        for w in range(outsize[0]):

            pattern = get_pattern(params, input_image_[w, h])
            npattern = len(pattern)
            pattern_ = pattern[np.mod(w, npattern)]

            wstart = w * pixsize[0] + np.mod(h, pixshift) * shift
            hstart = h * pixsize[1]

            fullimg.paste(pattern_, box=(wstart, hstart))

    return fullimg

def masking(img, facsize, alpha):
    level = int(255 * alpha)
    mask = Image.new('L', img.size, (level))
    draw = ImageDraw.Draw(mask)

    w, h = img.size
    ref = max(w, h)
    dd = (ref*facsize)*0.5
    xy = w*0.5-dd, h*0.5-dd, w*0.5+dd, h*0.5+dd
    draw.ellipse(xy, fill='white')
    img.putalpha(mask)

    return img


# - filtering

def get_slant_2D_gaussian_kernel(xstd=20, ystd=None, size=(80,80), angle=0):
    if len(size) == 1:
        xsize = size
        ysize = size
    else:
        xsize = size[0]
        ysize = size[1]

    if ystd is None:
        ystd = 0.1 * xstd

    kernel = Gaussian2DKernel( x_stddev=xstd, 
                               y_stddev=ystd, 
                               theta=(90 - angle) * pi/180, 
                               x_size=xsize, 
                               y_size=ysize, 
                               mode='center' )

    return np.array(kernel)

def get_slant_2D_gaussian_kernel_method1(xstd=20, ystd=None, size=(80,80), angle=0):
    if len(size) == 1:
        xsize = size
        ysize = size
    else:
        xsize = size[0]
        ysize = size[1]

    if ystd is None:
        ystd = 0.1 * xstd

    kernel = Gaussian2DKernel( x_stddev=xstd, 
                               y_stddev=ystd, 
                               theta=(90 - angle) * pi/180, 
                               x_size=xsize, 
                               y_size=ysize, 
                               mode='center' )

    return np.array(kernel)

def convolve(src, kernel, delta=None):
    src = cv.imread(src)

    if delta is None:
        output = cv.filter2D( src, -1, kernel )
    else:
        output = cv.filter2D( src, -1, kernel, delta=delta )

    return output

def convolve_method1(src, kernel, brightness=80, delta=None):
    src = cv.imread(src)
    src = cv.cvtColor(src, cv.COLOR_BGR2RGB)
    ksum = np.sum( kernel.ravel() )
    kernel = kernel * (2.0  / ksum) * (brightness / 100)

    if delta is None:
        output = cv.filter2D( src, -1, kernel )
    else:
        output = cv.filter2D( src, -1, kernel, delta=delta )

    return output

def convolve_method2(src, kernel):
    src = cv.imread(src)
    output = src.copy()

    kmax = np.max( kernel.ravel() )
    kernel = kernel * (1 / kmax)

    kh, kw = kernel.shape
    hrh, hrw, band = src.shape
    lrh, lrw  = int(np.round(hrh / kh)), int(np.round(hrw / kw))

    for h in range(lrh):
        for w in range(lrw):
            for rgb in range(band):
                x1 = w * kw
                x2 = np.min([x1 + kw, hrw])
                y1 = h * kh
                y2 = np.min([y1 + kh, hrh])

                output[y1:y2, x1:x2, rgb] = output[y1:y2, x1:x2, rgb] * kernel

    return output

def RMSE(original, convolved):
    M, N, B = original.shape
    err = original - convolved
    # rmse = np.sqrt(np.sum( np.power(err, 2).ravel() ) / (M * N * B))
    rmse = np.sqrt( np.mean( np.power(err, 2).ravel() ) )
    rmse = 10 * np.log10( rmse )

    return rmse


# - saving image function

def saving_images(outimg, names):
    if not os.path.isdir('output'):
        os.mkdir('output')

    for img, name in zip(outimg, names):
        print('saving image as: %s ...' %name)
        img.save(name, 'PNG')
        print('done')

def saving_image(outimg, name):
    if not os.path.isdir('output'):
        os.mkdir('output')

    print('saving image as: %s ...' %name)
    outimg.save(name, 'PNG')
    print('done')


# - add borders

def borders_outside(image, border):

    # - initializations

    w, h = image.size
    w, h = w + border[0] + border[1], h + border[2] + border[3]

    # - add borders

    outimg = Image.new('RGB', (w, h), (0,0,0))
    outimg.paste(image, box=(border[0], border[2]))

    return outimg

def borders_inside(image, border):

    # - initializations

    draw = ImageDraw.Draw(image)
    w, h = image.size

    # - add borders

    # - left

    start = 0, 0
    end = border[0], h
    draw.rectangle((start, end), fill=(0,0,0))

    # - right

    start = w - border[1], 0
    end = w, h
    draw.rectangle((start, end), fill=(0,0,0))

    # - top

    start = 0, 0
    end = w, border[2]
    draw.rectangle((start, end), fill=(0,0,0))

    # - bottom

    start = 0, h - border[3]
    end = w, h
    draw.rectangle((start, end), fill=(0,0,0))

    return image

def add_borders(image, border, outside=False):

    # - add borders as outside boolean

    if outside:
        outimg = borders_outside(image, border)
    else:
        outimg = borders_inside(image, border)

    return outimg


# - simple image generation

def get_white_circle(size, covert, grey_level, border=None, outside=False):

    # - initializations

    w, h = size
    covert = covert / 100
    grey_level = int(255 * (1 - grey_level / 100))

    # - create the output image

    outimg = Image.new('RGB', (w, h), (grey_level, grey_level, grey_level))
    draw = ImageDraw.Draw(outimg)

    # - put the white circle

    wdd = (w * covert) * 0.5
    hdd = (h * covert) * 0.5
    xy = w*0.5-wdd, h*0.5-hdd, w*0.5+wdd, h*0.5+hdd
    draw.ellipse(xy, fill='white')

    # - add border edges

    if border is not None:
        outimg = add_borders(outimg, border, outside=outside)

    return outimg

def get_slant_lines(size, angle, pitch, grey_level, color='white', border=None, outside=False):

    # - initializations

    w, h = size
    m = tan(angle * pi / 180)
    grey_level = int(255 * (1 - grey_level / 100))

    # - create output image

    outimg = Image.new('RGB', (w, h), (grey_level, grey_level, grey_level))
    draw = ImageDraw.Draw(outimg)

    # - put the slant lines 

    for x1 in range(0, w, pitch):
        x2 = np.round(m * h + x1)
        draw.line((x1, 0) + (x2, h), fill=color)

    for x1 in range(0, -h-w, -pitch):
        x2 = np.round(m * h + x1)
        draw.line((x1, 0) + (x2, h), fill=color)

    # - add border edges

    if border is not None:
        outimg = add_borders(outimg, border, outside=outside)

    return outimg

def white_circle():

    def usage():
      print('usage: %s <image size: (w,h)> <covert percent> <grey level> <output name>' % sys.argv[0])
      print()
      sys.exit(-1)

    if len(sys.argv) != 6:
        usage()

    print('generating a white circle image...')

    # - inputs

    w, h = sys.argv[2].replace(' ', '').split(',')
    size = int(w), int(h)
    covert = float(sys.argv[3])
    grey_level = float(sys.argv[4])
    output_name = sys.argv[5]

    # - get the circle white image

    outimg = get_white_circle(size, covert, grey_level)

    # - saving image

    saving_image(outimg, output_name)

def slant_lines():
    def usage():
      print('usage: %s <image size: (w,h)> <angle,pitch> <background grey level> <output name>' % sys.argv[0])
      print()
      sys.exit(-1)

    if len(sys.argv) != 6:
        usage()

    print('generating a image of slant lines ...')

    # - inputs

    w, h = sys.argv[2].replace(' ', '').split(',')
    size = int(w), int(h)
    angle, pitch = sys.argv[3].replace(' ', '').split(',')
    angle, pitch = float(angle), int(pitch)
    grey_level = float(sys.argv[4])
    output_name = sys.argv[5]

    # - get the image with slant lines

    outimg = get_slant_lines(size, angle, pitch, grey_level)

    # - saving image

    saving_image(outimg, output_name)


# - load the high resolution image

def get_high_res_image(filename, nrows=None):

    # - parse and get the parameters

    params = get_params(filename)

    if nrows is not None:
        params = params[:nrows]

    # - loadind images

    outimg, outname = [], []
        
    for params_ in params:
        print(' ')
        print('loading the RGB image: %s...' %params_['intname'])

        outimg.append( load_image(params_) )
        outname.append( 'output/%s' %params_['outname'] )

    return outimg, outname

def high_res_image():

    # - command line validation

    def usage():
      print('usage: %s <input csv file>' % sys.argv[0])
      print()
      sys.exit(-1)

    if len(sys.argv) != 3:
      usage()

    # - getting inputs from command line

    filename = sys.argv[2]

    # - get high res images

    outimg, outname = get_high_res_image(filename)

    # - saving images

    saving_images(outimg, outname)
    


# - execution

if __name__ == "__main__":

    globals()[sys.argv[1]]()
    