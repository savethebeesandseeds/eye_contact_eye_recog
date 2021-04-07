import os
import re
import sys
import json
import subprocess
import hashlib, random

# proc = subprocess.Popen(["cat", "/etc/services"], stdout=subprocess.PIPE, shell=True)
# (out, err) = proc.communicate()
# print("program output (/etc/services):", out)

# proc = subprocess.Popen(["cat", "/etc/release"], stdout=subprocess.PIPE, shell=True)
# (out, err) = proc.communicate()
# print("program output (/etc/release):", out)
class CIPHER_MECH:
    def __init__(self, max_date):
        self.mdfive = (lambda x,msg: (x.update(msg),x.hexdigest())[1])
        self.valid_host_keys = ['img_recognition.count_eye_contact_with_camera_video.host.'+str(_).zfill(4) for _ in range(1000)]
        self.valid_host_md5_hash = self.mdfive(hashlib.md5(), self.valid_host_keys[random.randint(0,999)].encode('UTF-8'))
        self.valid_key_labels = ['img_recognition.count_eye_contact_with_camera_video.key.'+str(_).zfill(4) for _ in range(1000)]
        self.valid_key_md5_hashes = [self.mdfive(hashlib.md5(), _.encode('UTF-8')) for _ in self.valid_key_labels]
        self.valid_vendor_ids = ['1d6b:0003', '----:---']
        self.max_date = max_date
    def check_is_RPI(self):
        checks = {
            'cpuinfo': {
                'flag':all([_=='fpasimdevtstrmcrc32cpuid' for _ in self.get_cpuinfo(return_format='list') if _ != '']),
                'false.msg':'VALIDATION ERROR CODE (#0) check was unsuccessful, code running on unrecognized plataform (not RASPBERRY PI).',
                'true.msg':'VALIDATION ASSERT CODE (#0) check was successful, code running on recognized plataform (RASPBERRY PI).',
            },
            'hostnamectl' : {
                'flag': 'arm' in self.get_hostnamectl(return_format='dict')['architecture'], 
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
            if(all([checks[_]['flag'] for _ in checks.keys()]) and not value['flag']):
                print(value['false.msg'])
        return any([checks[_]['flag'] for _ in checks.keys()])
    def comunicate_with_key(self):
        
        return None
    def check_KEY(self):
        def secure_key_protocol(host_hash, key_hash):
            pass
        # device_flag = False
        # for key, value in self.get_devices(only_uuid=True).items():
        #     if(value['label'] == key_label):
        #         subject_device = value
        #         device_flag = True
        # if(not device_flag):
        #     return device_flag
        return True # FIXME
    
    def check_all_is_well(self, fast=False):
        import time
        import sys
        print("Setting system...______")
        def progressbar(it, prefix="", size=60, file=sys.stdout):
            count = len(it)
            def show(j):
                x = int(size*j/count)
                file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
                file.flush()
            show(0)
            for i, item in enumerate(it):
                yield item
                show(i+1)
            file.write("\n")
            file.flush()
        for i in progressbar(range(100), "Loading enviroment fabric: \t\t", 40):
            if(not fast):time.sleep(0.1)
        for i in progressbar(range(100), "Setting enviroment fabric factors: \t", 40):
            if(not fast):time.sleep(0.04)
        for i in progressbar(range(100), "Securing conection: \t\t\t", 40):
            if(not fast):time.sleep(0.01)
        for i in progressbar(range(100), "Cleaning workspace: \t\t\t", 40):
            if(not fast):time.sleep(0.004)
        for i in progressbar(range(100), "Building intern methods: \t\t", 40):
            if(not fast):time.sleep(0.01)
            if(i == 13 and not self.check_KEY()):
                print("\n")
                return False
            if(i == 76 and not self.check_is_RPI()):
                print("\n")
                return False
        for i in progressbar(range(100), "Factor out enviroment dinamics: \t", 40):
            if(not fast):time.sleep(0.05)
            if(i==72):
                import requests, json, datetime
                url = "http://worldclockapi.com/api/json/est/now"
                res = requests.get(url)
                if(res.status_code != 200):
                    print("\n>> ERROR: No internet conection...")
                    return False
                c_date = json.loads(res.text)['currentDateTime']
                c_date = datetime.datetime.strptime(c_date.split('T')[0], '%Y-%m-%d')
                if(c_date > self.max_date):
                    print("\n>> The license of this executable has pass, to extend it please contact@waajacu.com.")
                    return False
        print("System READY, system is valid...______")
        return True

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
            return {_:str(re.search(hostnamectl_dict[_]+'(.*)', hostnamectl_string)).replace(str(hostnamectl_dict[_]), '').strip() for _ in hostnamectl_dict.keys()}
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
