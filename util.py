from __future__ import print_function
import os, time

class GlobalLog:
    LOG_ROOT_FOLDER = 'log'

    def __init__(self):
        self.log_files = {}
        self.log_folder = GlobalLog.LOG_ROOT_FOLDER
        self.header = ''
        self.tag = ''
        if not os.path.exists(GlobalLog.LOG_ROOT_FOLDER):
            os.makedirs(GlobalLog.LOG_ROOT_FOLDER)

    def __del__(self):
        for key in self.log_files:
            self.log_files[key].close()

    def _init_log_file(self, tag):
        f = open(os.path.join(GlobalLog.LOG_ROOT_FOLDER, tag), 'w')
        self.log_files[tag] = f
        f.write(time.strftime("start time: %Y-%m-%d %H:%M:%S\n", time.localtime(int(time.time()))))
        # f.write('params: {}\n'.format(str(Params)))
        if self.header is not None:
            f.write('header: {}\n'.format(self.header))

    def set_tag(self, tag):
        self.tag = tag

    def set_header(self, header):
        self.header = header

    def _log(self, line, end='\n', print_to_stdout=True):
        tag = self.tag
        line = str(line)
        if tag not in self.log_files:
            self._init_log_file(tag)
        if print_to_stdout:
            print(line, end=end)
        self.log_files[tag].write('{}{}'.format(line, end))

    def log(self, *args, end='\n', print_to_stdout=True):
        tag = self.tag
        content = (str(i) for i in args)
        line = ' '.join(content)
        if tag not in self.log_files:
            self._init_log_file(tag)
        if print_to_stdout:
            print(line, end=end)
        self.log_files[tag].write('{}{}'.format(line, end))

    def close_all(self):
        for k, v in self.log_files.items():
            v.close()
        self.log_files.clear()

    def move_log_file(self, tag, move_folder_name):
        move_folder = os.path.join(GlobalLog.LOG_ROOT_FOLDER, move_folder_name)
        if not os.path.isdir(move_folder):
            os.makedirs(move_folder)
        log_file_name = os.path.join(GlobalLog.LOG_ROOT_FOLDER, tag)
        os.system('mv {} {}'.format(log_file_name, move_folder))

    def flush(self):
        for i in self.log_files.values():
            i.flush()


Log = GlobalLog()
Log.set_tag('stdout')

