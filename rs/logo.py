"""An additional module for evaluating and deploying models. 
"""

from pathlib import Path
import json
import os
import requests
import argparse
from datetime import datetime
import pytz

from val import run, compute_metrics

def update_metadata(conf_thres, opt, data_yaml):
    # 1. meta.json
    meta = { 
        'size': opt.img_size,
        'base': opt.weights,
        'epochs': opt.epochs,
        'threshold': conf_thres
    }
    save_folder = Path('__temp__')
    save_folder.mkdir(exist_ok=True)
    meta_file = save_folder/'meta.json'
    with open(meta_file, 'w') as f:
        f.write(json.dumps(meta))
    gs_job_dir = opt.job_dir
    os.system(f'gsutil cp {meta_file} {gs_job_dir}')
    print(f'Copied meta.json to {gs_job_dir}')

    # 2. Copy all .pt files because there are multiple models
    for p in sorted(Path('runs/train').rglob('best.pt')):
        model_name = p.parent.parent.name.replace('exp', 'best') + '.pt'
        os.system(f'gsutil cp {p} {gs_job_dir}{model_name}')
        print(f'Copied {p} to {gs_job_dir}{model_name}')
    
    # 3. data.yaml
    os.system(f'gsutil cp {data_yaml} {gs_job_dir}')
    print(f'Copied data.yaml to {gs_job_dir}')
    
    # 4. Copy training logs
    os.system('rm -rf runs/train/exp/weights')
    os.system(f'gsutil -m cp -r runs/train/exp {gs_job_dir}')
    print(f'Copied training logs to {gs_job_dir}')
    
def update_model_pointers(gs_job_dir, gs_model_pointers_dir):
    # model.txt
    save_folder = Path('__temp__')
    save_folder.mkdir(exist_ok=True)
    model_type = 'brands-general'
    model_file = save_folder/(model_type + '.txt')
    with open(model_file, 'w') as f:
        f.write(gs_job_dir[5:-1])
        
    # 2 files: 1 for current and 1 for versioning
    os.system(f'gsutil cp {model_file} {gs_model_pointers_dir}')
    t = datetime.now().astimezone(pytz.timezone('Europe/London')).strftime('%Y%m%d_%H%M%S')
    version_model_file = model_type + '-' + t + '.version'
    os.system(f'gsutil cp {model_file} {gs_model_pointers_dir}{version_model_file}')
    print(f'Copied model pointers to {gs_model_pointers_dir}')
    
def deploy(circle_ci_token):
    """Make circleci rebuild and return pipeline id.
    """
    data = {
        'parameters': { 'run_workflow_deploy': True },
        'branch': 'release-1'
    }
    
    response = requests.post(
        'https://circleci.com/api/v2/project/github/redsift/logo-detection/pipeline', 
        headers={
            'content-type': 'application/json',
            'Circle-Token': circle_ci_token,
        },
        data=json.dumps(data))
    
    print(f'Posted request to circleci with {response.status_code} status code response')
    return response.json()['id']

def run_test(yaml, model, conf, results, gs_job_dir=None, prefix='', img_size=1280, thres=None):
    """Run tests, save results, record important metrics and return status.
    """
    print(f'\nEVALUATE {yaml}', flush=True)
    perf = compute_metrics(yaml, model, imgsz=img_size, conf_thres=conf)
    save_folder = Path('__temp__')
    save_folder.mkdir(exist_ok=True)
    
    if 'mean_f1' in thres: # Normal test
        if 'correct' not in perf:
            return False
        
        metrics_df, confusion_df = perf['correct']
        metrics_file = save_folder/(prefix + 'metrics.csv')
        metrics_df.to_csv(metrics_file)
        confusion_file = save_folder/(prefix + 'confusions.csv')
        confusion_df.to_csv(confusion_file)

        # Record important metrics
        stats = results[prefix] = {}
        stats['mean_f1'] = metrics_df['f1'].mean()
        stats['highest_f1'] = metrics_df.sort_values('f1', ascending=False).iloc[:5]['f1'].to_dict()
        stats['lowest_f1'] = metrics_df.sort_values('f1', ascending=True).iloc[:5]['f1'].to_dict()
        stats['lower_precision_threshold'] = metrics_df[metrics_df['precision'] < thres['prec']]['precision'].sort_values().to_dict()
        stats['lower_recall_threshold'] = metrics_df[metrics_df['recall'] < thres['recall']]['recall'].sort_values().to_dict()

        print(prefix)
        print(f'  Mean of F1 for all classes: {stats["mean_f1"]:.2f}')
        print(f'  Number of classes with precisions lower than {thres["prec"]}: {len(stats["lower_precision_threshold"])}')
        print(f'  Number of classes with recalls lower than {thres["recall"]}: {len(stats["lower_recall_threshold"])}')
        
        if gs_job_dir:
            gs_job_dir = gs_job_dir + 'stats/'
            print('  Copy metrics and confusions files to gscloud')
            os.system(f'gsutil cp {metrics_file} {gs_job_dir}')
            os.system(f'gsutil cp {confusion_file} {gs_job_dir}')

        # Test fails if either mean f1 is lower than threshold or precision/recall of any class is lower than threshold
        return stats['mean_f1'] >= thres['mean_f1'] and not stats['lower_precision_threshold'] and not stats['lower_recall_threshold']
    else: # False positives test
        rate, preds_df = perf['incorrect']
        preds_file = save_folder/(prefix + 'false_preds.csv')
        preds_df.to_csv(preds_file, index=None)
    
         # Record important metrics
        stats = results[prefix] = {}
        stats['preds_per_image'] = rate
        print(prefix)
        print(f'  #false predictions per image: {rate:.2f}')
        
        if gs_job_dir:
            gs_job_dir = gs_job_dir + 'stats/'
            print('  Copy false predictions to gscloud')
            os.system(f'gsutil cp {preds_file} {gs_job_dir}')
            
        return rate <= thres['preds_per_image']
    
def evaluate(validation_yaml, unit_test_yaml, model, gs_job_dir=None, img_size=1280, config=None):
    """Evaluate the model against a number of tests.
    Return True if all tests pass.
    """
    print('\n==================== EVALUATING MODEL ===================\n')
    
    # 1a. Compute validation mAP
    # 1b. Retrieve optimal confidence threshold for following metrics calculation and inference usage
    print(f'EVALUATE validation set {validation_yaml}', flush=True)
    val_map = run(validation_yaml, model, imgsz=img_size, get_map_alone=True)
    conf = config['conf_thres']
    val_status = val_map >= config['val_map_thres']
    results = { "mean_map_validation": val_map }
    print(f'validation mAP: {val_map:.2f}')
    print('val_status', val_status)
    
    # 2. Unit tests
    unit_test_status = run_test(unit_test_yaml, model, conf=conf, results=results, gs_job_dir=gs_job_dir, 
                                prefix='unit_test_', img_size=img_size, thres=config['unit_test'])
    print('unit_test_status', unit_test_status)
    
    # 3. Manual tests
    manual_tests_statuses = []
    for k, v in config['tests'].items():
        test_yaml = 'data/' + v['dir'] + '/data.yaml'
        _update_static_yaml(unit_test_yaml, test_yaml)
        status = run_test(test_yaml, model, conf=conf, results=results, gs_job_dir=gs_job_dir, 
                          prefix=k+'_', img_size=img_size, thres=v)
        manual_tests_statuses.append(status)
        print(k + '_status', status)

    # Write results
    save_folder = Path('__temp__')
    save_folder.mkdir(exist_ok=True)
    results_file = save_folder/'results.json'
    with open(results_file, 'w') as f:
        f.write(json.dumps(results))
        
    if gs_job_dir:
        print('Copy results file to gscloud')
        os.system(f'gsutil cp {results_file} {gs_job_dir}')
        
    if all([val_status, unit_test_status] + manual_tests_statuses):
        print('All tests suceeded')
        return True
    else:
        print('Tests failed; see results.json for more details')
        return False

def _update_static_yaml(source_file, dest_file):
    """handmade/false positive tests are static with a small number of classes. 
    However the classes in its yaml file should be the same as in the unittests 
    to avoid index error in prediction"""
    with open(source_file) as f:
        source_lines = f.readlines()
    with open(dest_file) as f:
        dest_lines = f.readlines()

    # The last 2 lines are nc and classes
    dest_lines[-2:] = source_lines[-2:]
    with open(dest_file, 'w') as f:
        f.write(''.join(dest_lines))
    
def main(opt, config):
    # 0. Start with updating the status
    update_status('running\nevaluation', opt.job_dir)

    # 1. Evaluate
    validation_yaml = 'data/' + opt.train_folder + '/data.yaml'
    unit_test_yaml = 'data/' + opt.unittest_folder + '/data.yaml'
    model = 'runs/train/exp/weights/best.pt'

    passed = False
    try:
        passed = evaluate(validation_yaml, unit_test_yaml, model, 
                        gs_job_dir=opt.job_dir, config=config)
    except Exception as e:
        print(f'Evaluation failed with exception.')

    # 2. Update training results
    update_metadata(config['conf_thres'], opt, validation_yaml)
    
    # 3. Deploy
    if not passed:
        print('Training finished with poor results. No deployment.')
        update_status('failed\nevaluation', opt.job_dir)
        return
    
    if config['deploy_after_train']:
        update_model_pointers(opt.job_dir, config['storage']['gs_model_pointers_dir'])
        pipeline_id = deploy(opt.circle_ci_token)
        update_status('running\ndeployment\n' + pipeline_id, opt.job_dir)
    else:
        print('Skip deploying as deploy_after_train=False')
        update_status('succeeded\nevaluation', opt.job_dir)
    
def load_configs(gs_params_file):
    os.system(f'gsutil cp {gs_params_file} params.json')
    with open('params.json') as f:
        return json.load(f)

def update_status(content, gs_job_dir):
    """Update status in to a 'status' file in the GS job dir.
    """
    with open('tmp_file', 'w') as f:
        f.write(content)
    os.system(f'gsutil cp tmp_file {gs_job_dir}status')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', type=str, required=True, help='GCS location for writing checkpoints and exporting models')
    parser.add_argument('--circle-ci-token', type=str, required=True, help='to deploy in circleci')
    parser.add_argument('--train-folder', type=str, required=True, help='name of the training data folder')
    parser.add_argument('--unittest-folder', type=str, required=True, help='name of the unit test data folder')
    parser.add_argument('--params-file', type=str, required=True, help='params.json config file')
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--img-size', type=int, default=640, help='image sizes')
    opt = parser.parse_known_args()[0]
    config = load_configs(opt.params_file)

    main(opt, config)