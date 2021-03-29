import os
import re
import sys
import json
import subprocess

# proc = subprocess.Popen(["cat", "/etc/services"], stdout=subprocess.PIPE, shell=True)
# (out, err) = proc.communicate()
# print("program output (/etc/services):", out)

# proc = subprocess.Popen(["cat", "/etc/release"], stdout=subprocess.PIPE, shell=True)
# (out, err) = proc.communicate()
# print("program output (/etc/release):", out)
class CIPHER_MECH:
    def __init__(self):
        pass
    def check_is_RPI(self):
        checks = {
            'cpuinfo': {
                'flag':all([_=='fpasimdevtstrmcrc32cpuid' for _ in self.get_cpuinfo(return_format='list') if _ != '']),
                'false.msg':'VALIDATION ERROR CODE (#0) check was unsuccessful, code running on unrecognized plataform (not RASPBERRY PI).',
                'true.msg':'VALIDATION ASSERT CODE (#0) check was successful, code running on recognized plataform (RASPBERRY PI).',
            },
            'hostnamectl' : {
                'flag': self.get_hostnamectl(return_format='dict')['architecture'] == 'arm64', 
                'false.msg':'VALIDATION ERROR CODE (#1) check was unsuccessful, code running on unrecognized plataform (not RASPBERRY PI).',
                'true.msg':'VALIDATION ASSERT CODE (#1) check was successful, code running on recognized plataform (RASPBERRY PI).',
            },
            'uname' : {
                'flag': all([self.get_uname()[_[0]] == _[1] for _ in [
                    ('kernel_name', 'Linux'), 
                    ('machine', 'aarch64'), 
                    ('processor', 'aarch64'), 
                    ('hardware_plataform', 'aarch64')
                    ]]), 
                'false.msg':'VALIDATION ERROR CODE (#2) check was unsuccessful, code running on unrecognized plataform (not RASPBERRY PI).',
                'true.msg':'VALIDATION ASSERT CODE (#2) check was successful, code running on recognized plataform (RASPBERRY PI).',
            },
        }
        for key, value in checks.items():
            if(not value['flag']):
                print(value['false.msg'])
        return all([checks[_]['flag'] for _ in checks.keys()])
    def check_KEY(self):
        def secure_key_protocol():
            pass
        key_label = 'waka'
        vendor_id = ['1d6b:0003', '----:---']
        device_flag = False
        for key, value in self.get_devices(only_uuid=True).items():
            if(value['label'] == key_label):
                subject_device = value
                device_flag = True
                
                
        return device_flag
    def get_lsusb(self, _vendor):
        # aux_list = [_.split('\t') for _ in os.popen('lsusb --verbose').read().split('\n')]
        aux_list = os.popen('lsusb -d {}'.format(_vendor)).read().split('\n')
        # print({'key':repr(aux_list[10][0])})
        return [_ for _ in aux_list if _ != '']#json.dumps(aux_list, indent=4)
    def get_cpuinfo(self, return_format='list'):
        if(return_format=='list'):
            aux_str = os.popen('cat /proc/cpuinfo | grep Features').read()
            return [str(_.replace(' ', '').replace('Features','').replace(':','').replace('\t','')) for _ in aux_str.split('\n')]
        else:
            return os.popen('cat /proc/cpuinfo').read()
    def get_active_services(self):
        return os.popen('cat /etc/services').read()
    def get_os_release(self):
        # grep '^VERSION' /etc/os-release
        return os.popen('cat /etc/os-release').read()
    def get_uname(self):
        uname_dict = {
            'kernel_name':'s',
            'node_name':'n',
            'kernel_release':'r',
            'kernel_version':'v',
            'machine':'m',
            'processor':'p',
            'hardware_plataform':'i',
            'operating_system':'o'
            }
        return {_:os.popen('uname -{}'.format(uname_dict[_])).read().replace('\n','') for _ in uname_dict.keys()}
    def get_hostnamectl(self, return_format='dict'):
        if(return_format=='dict'):
            hostnamectl_dict = {
                'static_hostname':r'Static hostname:',
                'icon_name':r'Icon name:',
                'chassis':r'Chassis:',
                'machine_id':r'Machine ID:',
                'boot_id':r'Boot ID:',
                'operating_system':r'Operating System:',
                'cpe_os_name':r'CPE OS Name:',
                'kernel':r'Kernel:',
                'architecture':r'Architecture:',
            }
            hostnamectl_string = os.popen('hostnamectl').read()
            return {_:str(re.search(hostnamectl_dict[_]+'(.*)', hostnamectl_string).group()).replace(str(hostnamectl_dict[_]), '').strip() for _ in hostnamectl_dict.keys()}
        else:
            return os.popen('hostnamectl').read()
    def get_devices(self, only_uuid=False):
        if(only_uuid):
            return [{'name':_['name'], 'label':_['label'], 'uuid':_['uuid'], 'ptuuid':_['ptuuid'], 'serial':_['serial']} for _ in json.loads(os.popen('lsblk -JO').read())['blockdevices']]
        else:
            return json.loads(os.popen('lsblk -JO').read())
if __name__=='__main__':
    cipher = CIPHER_MECH()
    # print(cipher.get_devices(only_uuid=True))
    # print(cipher.get_uname())
    # print(cipher.get_hostnamectl(return_format='dict'))
    # print(cipher.get_cpuinfo(return_format='list'))
    # print("check_is_RPI: {}".format(cipher.check_is_RPI()))
    print(cipher.get_lsusb(_vendor='1d6b:0003'))
