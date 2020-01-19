import cv2
import numpy as np

def read_image(img_path):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if not img.dtype == np.uint8:
        pass

    return img

def write_image(img, img_saving_path):
    """Writes an image to a given path.
    """
    if isinstance(img, np.ndarray):
        if not img.dtype == np.uint8:
            assert np.max(img) <= 1, "Maximum pixel value {:.3f} is greater than 1".format(np.max(img))
            img = (255 * img).astype(np.uint8)
    else:
        raise TypeError("img is not a ndarray.")
    
    cv2.imwrite(img_saving_path, img)

def zero_pad(img, pwx, pwy):
    """Pads a given image with zero at the border.
    """
    padded_img = np.zeros((img.shape[0]+2*pwy, img.shape[1]+2*pwx))
    for i in range(0, img.shape[0]):
        padded_img[pwy+i][pwx: img.shape[1]+pwx] = np.copy(img[i][:])
    return padded_img     