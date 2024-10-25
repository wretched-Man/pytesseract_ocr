# We are going to use Google translate on
# the images we extracted in Thayer's Lexicon

# We extract Greek text using re
import glob
import re
from googletrans import Translator
import cv2
from tess_ocr import pre_process, tesseract_ocr

# make translations
translator = Translator()

def greek_to_english(match_string):
    translation = translator.translate(match_string.group(), dest='en')
    #print(type('translation.text'))
    return translation.text

# capture all greek text

greek = re.compile(r'\w{1,}[^a-z0-9\s.,-;”)@\']{2}')

# read in the images
image_names = sorted(glob.glob("./thayer_pdfs/*"))
filepath = "out_trans/"

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
        # find greek text
        new_string = greek.sub(greek_to_english, string)
        base_name = name.split('/')[-1].split('.')[0]
        filename = filepath + base_name.lower().replace(' ', '_')
        with open(filename+'.txt', "w") as f:
            f.write(new_string)
