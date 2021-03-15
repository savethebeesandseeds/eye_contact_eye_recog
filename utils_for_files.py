import os
import shutil

def assert_file(path_, assert_by_creation=False):
    if(os.path.isfile(path_)):
        return True
    elif(assert_by_creation):
        str_aux = 'assert_by_creation error, not implemented'
        assert False, str_aux
    else:
        aux_str = 'assert_file error, File <{}> does not exist.'.format(path_)
        assert False, aux_str
    return False

def reset_folder(path_, just_content=False):
    print("Reseting folder: {}".format(path_))
    if(not os.path.isdir(path_) or not just_content):
        if(os.path.isdir(path_)):
            shutil.rmtree(path_)
        os.makedirs(path_)
    else:
        for pth in os.listdir(path_):
            if(os.path.isfile(pth)):
                os.remove(pth)
            elif(os.path.isdir(pth)):
                shutil.rmtree(pth)
