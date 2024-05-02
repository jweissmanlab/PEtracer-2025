import sys
import numpy as np
import skimage as ski
import pandas as pd
import scipy as sp
import xml


sys.path.append('/lab/weissman_imaging/puzheng/Softwares/')
from ChromAn.src.file_io import dax_process

def get_dax_attributes(dax_path):
    xml_path = dax_path.replace(".dax", ".xml")
    attrs = {}
    root = xml.etree.ElementTree.parse(xml_path).getroot()
    stage_position = root.find('.//acquisition/stage_position').text
    attrs["stage_position"] = np.array([float(i) for i in stage_position.split(",")])
    objective = root.find('.//mosaic/objective').text
    micron_per_pixel = root.find(f'.//mosaic/{objective}').text
    attrs["micron_per_pixel"] = float(micron_per_pixel.split(",")[1])
    attrs["flip_horizontal"] = root.find('.//mosaic/flip_horizontal').text == "True"
    attrs["flip_vertical"] = root.find('.//mosaic/flip_vertical').text == "True"
    attrs["transpose"] = root.find('.//mosaic/transpose').text == "True"
    return attrs

def load_fov(fov,rounds,img_path,file_pattern,z_project = False,z_slices = None, ref_index = 0):
    imgs = []
    attrs = []
    channels = rounds["color"].nunique()
    for series, series_rounds in rounds.groupby("series",sort = False):
        if file_pattern.split(".")[-1] == "dax":
            img_file = img_path + file_pattern.format(series=series,fov=fov)
            attr = get_dax_attributes(img_file)
            daxp = dax_process.DaxProcesser(img_file)
            daxp._load_image()
            for color in series_rounds["color"]:
                img = getattr(daxp,"im_"+str(color))
                if z_slices != None:
                    img = img[z_slices,:,:]
                if z_project:
                    img = img.max(axis = 0)
                if attr["transpose"]:
                    img = img.swapaxes(-1,-2)
                if attr["flip_horizontal"]:
                    img = np.flip(img, axis=-1)
                if attr["flip_vertical"]:
                    img = np.flip(img, axis=-2)
                imgs.append(img)
                attrs.append(attr)
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
    if len(attrs) > 0:
        return imgs, attrs[ref_index]
    else:
        return imgs, None

def unsharp_mask(img, filter_sigma, channel_axis = None):
    blur = ski.filters.gaussian(img, sigma=filter_sigma, channel_axis = channel_axis,truncate=2,preserve_range=True)
    img = np.clip(img - blur,0,np.iinfo(img.dtype).max).astype(np.uint16)
    return img

def get_rgb(imgs,thresholds,axis = None):
    rgb = []
    for img, threshold in zip(imgs,thresholds):
        if axis:
            img = img.max(axis = axis)
        img = np.clip(img / np.percentile(img ,threshold),0,1)
        rgb.append(img)
    return np.stack([rgb[0],rgb[1],np.clip(rgb[0] + rgb[2],0,1)],axis = -1) 