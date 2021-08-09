import os
import argparse
from pathlib import Path

def download_bucket(gs_root_folder, folder_name):
    print(f'DOWNLOAD {folder_name}', flush=True)
    tmp_zip_file = 'tmp_file.zip'
    data_path = gs_root_folder + '/' + folder_name + '.zip'
    os.system(f'gsutil cp {data_path} {tmp_zip_file}')
    os.system(f'unzip -qo {tmp_zip_file} -d data && rm {tmp_zip_file}')
    
def download(gs_root_folder, train_folder, unittest_folder, handmade_folder):
    """Download data from gs bucket to `data` folder, unzip and delete the ziped files.
    """
    print('\n==================== DOWNLOADING DATA ===================\n')
    download_bucket(gs_root_folder, train_folder)
    download_bucket(gs_root_folder, unittest_folder)
    download_bucket(gs_root_folder, handmade_folder)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gs-root-folder', type=str, default='gs://logo-training/data', help='gs bucket to the root data directory')
    parser.add_argument('--train-folder', type=str, required=True, help='name of the training data folder')
    parser.add_argument('--unittest-folder', type=str, required=True, help='name of the unit test data folder')
    parser.add_argument('--handmade-folder', type=str, default='handmade', help='name of the handmade test data folder')
    opt = parser.parse_known_args()[0]
    download(opt.gs_root_folder, opt.train_folder, opt.unittest_folder, opt.handmade_folder)