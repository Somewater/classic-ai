#!/usr/bin/env python3

import os
import zipfile
from datetime import datetime

now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
root_path = os.path.dirname(os.path.abspath(__file__))

def zip_files(zip, source_dir, files_pattern = '*', exclude_pattern = '__pycache__'):
    relroot = os.path.abspath(os.path.join(source_dir, os.pardir))
    for root, dirs, files in os.walk(source_dir):
        zip.write(root, os.path.relpath(root, relroot))
        for file in files:
            filename = os.path.join(root, file)
            if os.path.isfile(filename):
                if (files_pattern == '*' or file in files_pattern) and not (exclude_pattern in filename):
                    arcname = os.path.join(os.path.relpath(root, relroot), file)
                    zip.write(filename, arcname)
                    print('WRITE %s' % filename)

if __name__ == '__main__':
    output_filename = os.path.join(root_path, 'my-%s.zip' % now)
    with zipfile.ZipFile(output_filename, "w", zipfile.ZIP_DEFLATED) as zip:
        zip_files(zip, os.path.join(root_path, 'my'), '*')
        zip_files(zip, os.path.join(root_path, 'data'), ['stop_words.csv', 'ortho.pickle',
                                                         'frequency.pickle', 'frequency_tree.bin',
                                                         'frequency_lemms.pickle', 'frequency_tree_lemms.bin'])
        zip_files(zip, os.path.join(root_path, 'weights/5-2-3-4e-05-1-1_1M'))
        zip.write(os.path.join(root_path, 'my', 'metadata.json'), 'metadata.json')
        zip.write(os.path.join(root_path, 'server.py'), 'server.py')