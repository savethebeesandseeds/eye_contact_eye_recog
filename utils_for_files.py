# This code is propietary of www.waajacu.com
# was developed by santiago restrepo.
import os
import shutil
from datetime import datetime

def get_date(_format='%Y-%m-%d %H-%M-%S'):
    return str(datetime.today().strftime(_format))

def assert_file(_path, _label='', write_init=True):
    if(not os.path.isfile(_path)):
        with open(_path, "w+") as f:
            if(write_init):
                f.write("{} :: Initlializing <{}> file: ".format(get_date(), _label))
    aux_str = 'please specify an appropriate file path for <{}> as <{}> is unrecognized or badly defined.'.format(_label, _path)
    assert os.path.isfile(_path), aux_str

def update_file(_path, _label='', _data='nothing to be said', append_flag=False):
    assert_file(_path, _label='', write_init=False)
    if(append_flag):
        with open(_path, "a+") as f:
            f.write("{} :: <{}> :: {}".format(get_date(), _label, _data))
    else:
        with open(_path, "w+") as f:
            f.write("{} :: <{}> :: {}".format(get_date(), _label, _data))


def assert_file_(path_, assert_by_creation=False):
    if(os.path.isfile(path_)):
        return True
    elif(assert_by_creation):
        str_aux = 'assert_by_creation error, not implemented'
        assert False, str_aux
    else:
        aux_str = 'assert_file error, File <{}> does not exist.'.format(path_)
        assert False, aux_str
    return False

def assert_folder(path_):
    if(not os.path.isdir(path_)):
        os.makedirs(path_)

def reset_folder(path_, just_content=False):
    print("Reseting folder: {}".format(path_))
    if(not os.path.isdir(path_) or not just_content):
        if(os.path.isdir(path_)):
            shutil.rmtree(path_)
        os.makedirs(path_)
    else:
        for pth in os.listdir(path_):
            aux_pth = os.path.normpath(os.path.join(path_,pth))
            if(os.path.isfile(aux_pth)):
                os.remove(aux_pth)
            elif(os.path.isdir(aux_pth)):
                shutil.rmtree(aux_pth)
