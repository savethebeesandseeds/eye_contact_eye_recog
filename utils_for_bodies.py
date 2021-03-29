# This code is propietary of www.waajacu.com
# was developed by santiago restrepo.
import os
import dlib
import uuid
from PIL import Image

def object_candidates(image_file, temp_folder):
    img = dlib.load_rgb_image(image_file)

    # Locations of candidate objects will be saved into rects
    rects = []
    dlib.find_candidate_object_locations(img, rects, min_size=50000)

    print("number of rectangles found {}".format(len(rects))) 
    for k, d in enumerate(rects):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        aux_image = img[d.top():d.bottom(), d.left():d.right()]
        pil_image = Image.fromarray(aux_image)
        # ---
        img_path = os.path.normpath(os.path.join(temp_folder, '{}.jpg'.format(str(uuid.uuid1()))))
        pil_image.save(img_path)
