import os

def train(weights_reuse=False, half=False, freeze=False):
    data_file = 'data/selected13-100x-v4-half/data.yaml' if half else 'data/selected13-100x-v4-pure/data.yaml'
    weights = 'runs/train/selected10-100x-v4-pure/weights/best.pt' if weights_reuse else 'yolov5l6.pt'
    model_name = f'w{weights_reuse}-d{half}-l{freeze}-40'
    cmd = f'python train.py --data {data_file} --weights {weights} --name {model_name} --batch 6 --epochs 40 --img-size 1280'
    if freeze:
        cmd += ' --freeze'
    os.system(cmd)

if __name__ == "__main__":
    for weights_reuse in [False, True]:
        for half in [False, True]:
            for freeze in [False, True]:
                train(weights_reuse, half, freeze)