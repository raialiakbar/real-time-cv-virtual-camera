# -*- coding: utf-8 -*-
"""
Created on Mon May  3 19:18:29 2021

@author: droes
"""
from numba import njit  # conda install numba
import numpy as np
import cv2  # used by equalize_image (NOT inside the @njit function)

@njit
def histogram_figure_numba(np_img):
    '''
    Jit compiled function to increase performance.
    Use some loops insteads of purely numpy functions.
    If you face some compile errors using @njit, see: https://numba.pydata.org/numba-doc/dev/reference/numpysupported.html
    In case you dont need performance boosts, remove the njit flag above the function
    Do not use cv2 functions together with @njit
    '''
    
    # return r_bars, g_bars, b_bars
    height, width, _ = np_img.shape

    # One bin per possible 8-bit intensity (0..255), per channel.
    r_bars = np.zeros(256, dtype=np.float64)
    g_bars = np.zeros(256, dtype=np.float64)
    b_bars = np.zeros(256, dtype=np.float64)

    # Count how many pixels fall at each intensity, per channel.
    # Frame is RGB (run.py uses bgr_to_rgb=True): channel 0=R, 1=G, 2=B.
    for y in range(height):
        for x in range(width):
            r_bars[np_img[y, x, 0]] += 1.0
            g_bars[np_img[y, x, 1]] += 1.0
            b_bars[np_img[y, x, 2]] += 1.0

    # Scale each channel so its tallest bar maps to 3.0 (the plot's y-limit).
    # Guard against divide-by-zero on an all-black channel.
    r_max = r_bars.max()
    g_max = g_bars.max()
    b_max = b_bars.max()
    if r_max > 0:
        r_bars = r_bars / r_max * 3.0
    if g_max > 0:
        g_bars = g_bars / g_max * 3.0
    if b_max > 0:
        b_bars = b_bars / b_max * 3.0

    return r_bars, g_bars, b_bars



####

### All other basic functions

def image_stats(np_img):
    '''
    Computes per-channel (R, G, B) basic statistics:
    mean, mode, standard deviation, min and max.
    Returns a list of formatted strings ready for plot_strings_to_image().
    np_img is expected to be an RGB uint8 image.
    '''
    channel_names = ('R', 'G', 'B')
    lines = []
    for c, name in enumerate(channel_names):
        ch = np_img[:, :, c]
        mean_val = float(ch.mean())
        std_val = float(ch.std())
        min_val = int(ch.min())
        max_val = int(ch.max())
        # mode = the most frequently occurring intensity value in the channel
        counts = np.bincount(ch.ravel(), minlength=256)
        mode_val = int(counts.argmax())
        lines.append(
            f'{name} mean:{mean_val:.0f} mode:{mode_val} '
            f'std:{std_val:.0f} min:{min_val} max:{max_val}'
        )
    return lines


def equalize_image(np_img):
    '''
    Histogram equalization using cv2.equalizeHist, applied per RGB channel.
    Note: equalizing channels independently can shift colors slightly. A
    common alternative is to convert to YCrCb and equalize only the Y
    (luminance) channel, which preserves color better. We keep the simple
    per-channel version here.
    np_img: RGB uint8 -> returns RGB uint8.
    '''
    # cv2 needs a contiguous array; sequence may be a negative-stride view.
    np_img = np.ascontiguousarray(np_img)
    channels = cv2.split(np_img)
    equalized = [cv2.equalizeHist(ch) for ch in channels]
    return cv2.merge(equalized)

def linear_transform(np_img, a =1.3, b=25):
    '''
    This is a linear brightness/contrast transformation, using the formula:
    output = a * input + b
    putting an alpha value above 1 increases contrast, where a beta of more
    than 0 increases brightness
    '''
    transformed = a * np_img.astype(np.float32) + b
    transformed = np.clip(transformed, 0, 255)
    return transformed.astype(np.uint8)
    
def entropy_per_channel(np_img):
    '''
    This function computes the entropy separately for each of the RGB channels
    Entropy tells us how much information is in the image, with higher values
    indicating more detail and/or variation
    '''
    channel_names = ('R', 'G', 'B')
    lines = []
    
    for c, name in enumerate(channel_names):
        ch = np_img[:, :, c]
        
        counts = np.bincount(ch.ravel(), minlength=256).astype(np.float64)
        probs = counts / counts.sum()
        
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))
        
        lines.append(f'{name} entropy:{entropy:.2f}')
        
    return lines

def gaussian_blur(np_img, kernel_size=9):
    '''
    This function applies a Gaussian blur filter
    '''
    np_img = np.ascontiguousarray(np_img)
    return cv2.GaussianBlur(np_img, (kernel_size, kernel_size), 0)

def sobel_edges(np_img):
    '''
    This function applies Sobel Edge Detection and returns
    RGB image so pyvirtualcam still receives 3 channels
    '''
    np_img = np.ascontiguousarray(np_img)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    return cv2.cvtColor(magnitude, cv2.COLOR_GRAY2RGB)
    
####