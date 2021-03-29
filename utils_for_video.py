# This code is propietary of www.waajacu.com
# was developed by santiago restrepo.
import cv2
import uuid
from time import time, sleep
class video_mech:
    def __init__(self, temp_folder, process_at_fps=1, source='cam'):
        # source == is the path
        self.temp_folder = temp_folder
        if(source=='cam'):
            assert False, 'no cam configured'
            self.get_frame = self.get_frame_cam
        else:
            self.source = source
            self.get_frame = self.get_frame_video
            self.vidcap = cv2.VideoCapture(self.source)
            self.total_frames = int(self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.source_fps = int(self.vidcap.get(cv2.CAP_PROP_FPS))
            self.fn_cnt = 0
            assert self.source_fps >= process_at_fps, 'processing fps configured is too high'
            self.frame_jump = int(self.source_fps/process_at_fps)
    def get_frame_cam(self):
        assert False, 'not yet implemented'
    def get_frame_video(self, frame_jump=None, idx=None):
        if(frame_jump is not None):
            self.fn_cnt += frame_jump
        # p_start = time()
        self.vidcap.set(cv2.CAP_PROP_POS_FRAMES, self.fn_cnt)
        # print("execution time to set vidcap: {}".format(time()-p_start))
        # p_start = time()
        success, self.c_image = self.vidcap.read()
        # print("execution time to read vidcap: {}".format(time()-p_start))
        # print('Read a new frame #{}: '.format(self.fn_cnt), success)
        self.fn_cnt += self.frame_jump
        if(idx is None):
            self.image_id = str(uuid.uuid1())
        else:
            self.image_id = str(idx)+'_____'+str(uuid.uuid1())
        return success
    def save_c_image(self, path_=None):
        if(path_ is not None):
            assert False, 'not configured'
        else:
            img_temp_path = self.temp_folder+"/{}___{}.jpg".format(self.image_id, self.fn_cnt)
        cv2.imwrite(img_temp_path, self.c_image)
        return img_temp_path, self.image_id
