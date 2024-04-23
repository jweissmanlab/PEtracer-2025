import sys
import numpy as np
import skimage as ski
import pandas as pd
import scipy as sp


sys.path.append('/lab/weissman_imaging/puzheng/Softwares/')
from ChromAn.src.file_io import dax_process

def load_fov(fov,rounds,img_path,file_pattern,z_project = False,z_slices = None):
    imgs = []
    channels = rounds["color"].nunique()
    for series, series_rounds in rounds.groupby("series",sort = False):
        if file_pattern.split(".")[-1] == "dax":
            daxp = dax_process.DaxProcesser(img_path + file_pattern.format(series=series,fov=fov))
            daxp._load_image()
            for color in series_rounds["color"]:
                img = getattr(daxp,"im_"+str(color))
                if z_slices != None:
                    img = img[z_slices,:,:]
                if z_project:
                    img = img[z_slices,:,:].max(axis = 0)
                imgs.append(img)
        elif file_pattern.split(".")[-1] == "tif":
            img = ski.io.imread(img_path + file_pattern.format(series=series,fov=fov))
            img = img.reshape((channels, -1, img.shape[-2], img.shape[-1]), order='F')
            if z_slices != None:
                img = img[:,z_slices,:,:]
            if z_project:
                img = img.max(axis = 1)
            for color in img:
                imgs.append(color)
    imgs = np.stack(imgs,axis = 0)
    return imgs

def unsharp_mask(img, filter_sigma, channel_axis = None):
    blur = ski.filters.gaussian(img, sigma=filter_sigma, channel_axis = channel_axis,truncate=2,preserve_range=True)
    img = np.clip(img - blur,0,np.iinfo(img.dtype).max).astype(np.uint16)
    return img

def rgb_projection(imgs,thresholds,axis = 0):
    projected = []
    for img, threshold in zip(imgs,thresholds):
        img = img.max(axis = axis)
        img = np.clip(img / np.percentile(img ,threshold),0,1)
        projected.append(img)
    return np.stack([projected[0],projected[1],np.clip(projected[0] + projected[2],0,1)],axis = -1) 