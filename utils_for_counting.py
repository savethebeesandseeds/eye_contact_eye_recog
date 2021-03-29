# This code is propietary of www.waajacu.com
# was developed by santiago restrepo.
import os
import utils_for_files

class COUNTER_MECH:
    def __init__(self, report_folder, file_label, report_filename):
        self.file_label = file_label
        self.report_folder = report_folder
        self.report_file = os.path.join(self.report_folder, report_filename)
        utils_for_files.assert_folder(path_=self.report_folder)

    def update_report(self, _data=''):
        utils_for_files.update_file(_path=self.report_file, _label=self.file_label, _data=_data, append_flag=False)
        print(">> REPORT FILE UPDATED")

