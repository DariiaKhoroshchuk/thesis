import pydicom
import cv2


def read_img(path, apply_windowing=True):
    if '.dcm' in path:
        ds = pydicom.dcmread(path)
        img = ds.pixel_array
        if apply_windowing:
            img = pydicom.pixel_data_handlers.apply_windowing(img,ds)
    elif '.tif' in path:
        img = cv2.imread(path,0)
    else:
        img = cv2.imread(path)
    return img



