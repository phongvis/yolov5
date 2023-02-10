import os
import argparse
import json
from pathlib import Path

def download_bucket(gs_data_dir, folder_name):
    print(f'DOWNLOAD {folder_name}', flush=True)
    tmp_zip_file = 'tmp_file.zip'
    data_path = gs_data_dir + folder_name + '.zip'
    os.system(f'gsutil cp {data_path} {tmp_zip_file}')
    os.system(f'unzip -qo {tmp_zip_file} -d data && rm {tmp_zip_file}')
    
def download(gs_data_dir, train_folder, unittest_folder, manual_folders):
    """Download data from gs bucket to `data` folder, unzip and delete the ziped files.
    """
    print('\n==================== DOWNLOADING DATA ===================\n')
    download_bucket(gs_data_dir, train_folder)
    download_bucket(gs_data_dir, unittest_folder)
    for v in manual_folders.values():
        download_bucket(gs_data_dir, v['dir'])

def load_configs(gs_params_file):
    os.system(f'gsutil cp {gs_params_file} params.json')
    with open('params.json') as f:
        return json.load(f)

def update_status(gs_job_dir):
    """Update status in to a 'status' file in the GS job dir.
    """
    with open('tmp_file', 'w') as f:
        f.write('running')
    os.system(f'gsutil cp tmp_file {gs_job_dir}status')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-folder', type=str, required=True, help='name of the training data folder')
    parser.add_argument('--unittest-folder', type=str, required=True, help='name of the unit test data folder')
    parser.add_argument('--params-file', type=str, required=True, help='params.json config file')
    parser.add_argument('--job-dir', type=str, required=True, help='job directory')
    opt = parser.parse_known_args()[0]
    config = load_configs(opt.params_file)
    update_status(opt.job_dir)
    download(config['storage']['gs_data_dir'], opt.train_folder, opt.unittest_folder, config['tests'])