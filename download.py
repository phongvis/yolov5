import os
import argparse
from pathlib import Path

def download(bucket):
    """Download data from gs bucket to `data` folder, unzip and delete the ziped file."""
    print('DOWNLOAD DATA from', bucket)
    zip_file = 'data/data.zip'
    os.system(f'gsutil cp {bucket} {zip_file}')
    os.system(f'unzip -qo {zip_file} -d data && rm {zip_file}')
    
    dataset = Path(bucket).stem
    print('#training images:', len(list((Path('data')/dataset/'images/train').glob('*.jpg'))))
    print('#validation images:', len(list((Path('data')/dataset/'images/valid').glob('*.jpg'))))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gs-data', type=str, required=True, help='data.yaml path in gs')
    opt = parser.parse_known_args()[0]
    download(opt.gs_data)