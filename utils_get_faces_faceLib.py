# This code is propietary of www.waajacu.com
# was developed by santiago restrepo.
import os
import uuid
from PIL import Image
from face_recognition import face_locations, load_image_file

def get_faces(image, method="HOG"):
    # IMG_PATH: Path/to/image.jpg
    # METHOD: 'HOG'/'cnn'
    # Load the jpg file into a numpy array
    
    # Find all the faces in the image using a pre-trained convolutional neural network.
    # This method is more accurate than the default HOG model, but it's slower
    # unless you have an nvidia GPU and dlib compiled with CUDA extensions. But if you do,
    # this will use GPU acceleration and perform well.
    # See also: find_faces_in_picture.py
    face_loc = face_locations(image, number_of_times_to_upsample = 0, model = method) # HOG/cnn
    print("Found #{} face(s) in this photograph.".format(len(face_loc)))
    face_locations_to_pixel_corners(image, face_loc, save_flag = True)

def face_locations_to_pixel_corners(image, face_loc, save_flag = False):
    face_metada_data = {}
    face_metada_data['faces_count'] = len(face_loc)
    face_metada_data['faces'] = []
    for idx, face_location in enumerate(face_loc):
        face_metada_data['faces'].append({
            'id': str(uuid.uuid1()),
            'location': {
                'top':0, 
                'right':0, 
                'bottom':0, 
                'left':0,
                }
            })
        # Print the location of each face in this image
        top, right, bottom, left = face_location
        face_metada_data['faces'][idx]['location']['top'] = top
        face_metada_data['faces'][idx]['location']['right'] = right 
        face_metada_data['faces'][idx]['location']['bottom'] = bottom 
        face_metada_data['faces'][idx]['location']['left'] = left
        print("A face was found located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
        pil_image = sub_image(image=image, top=top, right=right, bottom=bottom, left=left, return_format = 'pil')
        if(save_flag):
            save_img_path = os.environ["TEMP_FOLDER"] + '/' + face_metada_data['faces'][idx]['id']+'.jpg'
            save_pill_image(pil_image, path_ = save_img_path)
def sub_image(image, top, right, bottom, left, return_format = 'array'):
    # You can access the actual face itself like this:
    face_image = image[top:bottom, left:right]
    # ---
    if(return_format == 'array'):
        return face_image
    elif(return_format == 'pil'):
        return Image.fromarray(face_image)

def save_pill_image(pil_image, path_):
    print("saving image to file: {}".format(path_))
    pil_image.save(path_)
        