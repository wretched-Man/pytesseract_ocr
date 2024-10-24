# Image OCR, V2,
# Input multiple images
import pytesseract
import cv2
import numpy as np


def pre_process(img_gray):
    """
        preprocesses an image, for input.
        The output here is the bbox of the largest contour.

        Expects a grayscale image.
    """
    # threshold and denoise
    img_thresh = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 0)

    # we will only OCR the largest contour in a page
    contours, _ = cv2.findContours(
        img_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sort_cont = sorted(contours, key=cv2.contourArea, reverse=True)
    # multiply by 4, since we scaled down input by 4
    # implementation specific, since images were large.
    # remove if that's not the case
    rect = cv2.boundingRect(sort_cont[0]) * np.full(4, 4)
    return rect


def tesseract_ocr(img):
    # use LSTM
    string = pytesseract.image_to_string(
        img, config="-l eng+grc --psm 6 --oem 1")

    left = ''
    right = ''
    lines = string.split('\n')

    for line in lines:
        if "|" in line:
            right += line.split("|")[-1]
            for splitted in line.split("|")[:-1]:
                left += splitted
        elif "-" in line:
            right += line.split("-")[-1]
            for splitted in line.split("|")[:-1]:
                left += splitted
        else:
            left += line
    if len(right) == 0:
        return left
    # add a page separator
    left += "\n\n --RIGHT-- \n\n"
    return left+right

def main():
    import glob
    image_names = sorted(glob.glob("./thayer_pdfs/*"))

    filepath = "output_text/"

    for name in image_names:
        img_cv = cv2.imread(name)
        img_resize = cv2.resize(img_cv, None, None, .25, .25, cv2.INTER_AREA)
        img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)

        # pre process
        x, y, w, h = pre_process(img_gray)

        # extract main contour
        main_contour = img_cv[y:y+h, x:x+w].copy()

        string = tesseract_ocr(main_contour)

        if len(string) > 0:
            base_name = name.split('/')[-1].split('.')[0]
            filename = filepath + base_name.lower().replace(' ', '_')
            with open(filename+'.txt', "w") as f:
                f.write(string)
    return 0

if __name__ == '__main__':
    main()