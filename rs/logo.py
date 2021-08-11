"""An additional module for evaluating and deploying models. 

Usage:
optimal_conf = evaluate(...)
if optimal_conf is not None:
    update_metadata(optimal_conf, model_details)
    deploy(...)
"""
# import sys
# if '../../' not in sys.path:
#     sys.path.append('../../')

from pathlib import Path
import json
import os
import requests
import argparse
from datetime import datetime
import pytz

# from root.val import run, compute_metrics
from val import run, compute_metrics

def update_metadata(optimal_conf, model_details, gs_job_dir, gs_model_pointers_dir):
    # 1. meta.json
    meta = { 
        'size': model_details['size'],
        'base': model_details['base'],
        'epochs': model_details['epochs'],
        'threshold': optimal_conf,
    }
    save_folder = Path('__temp__')
    save_folder.mkdir(exist_ok=True)
    meta_file = save_folder/'meta.json'
    with open(meta_file, 'w') as f:
        f.write(json.dumps(meta))
    os.system(f'gsutil cp {meta_file} {gs_job_dir}')
    print(f'Copied meta.json to {gs_job_dir}')
    
    # 2. best.pt
    os.system(f'gsutil cp {model_details["model"]} {gs_job_dir}best.pt')
    print(f'Copied best.pt to {gs_job_dir}')
    
    # model
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
    
    # Copy training logs
    os.system('rm -rf runs/train/exp/weights')
    os.system(f'gsutil cp -r runs/train/exp {gs_job_dir}')
    print(f'Copied training logs to {gs_job_dir}')

def deploy(circle_ci_token):
    """Make circleci rebuild.
    """
    print(f'Deployment with {circle_ci_token} is cancelled for testing')
    return 

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
    
    print(response.status_code)

def run_test(yaml, model, conf, results, gs_job_dir=None, prefix='', mean_f1_thres=0.5, prec_thres=0.5, recall_thres=0.5, img_size=1280):
    """Run tests, save results, record important metrics and return status.
    """
    print(f'EVALUATE {yaml}', flush=True)
    metrics = compute_metrics(yaml, model, imgsz=img_size, conf_thres=conf)
    if metrics is None:
        return True
        raise Exception('No correct predictions. There should be something wrong with the model.')
        
    metrics_df, confusion_df = metrics    
    save_folder = Path('__temp__')
    save_folder.mkdir(exist_ok=True)
    metrics_file = save_folder/(prefix + 'metrics.csv')
    metrics_df.to_csv(metrics_file)
    confusion_file = save_folder/(prefix + 'confusions.csv')
    confusion_df.to_csv(confusion_file)
    
    # Record important metrics
    stats = results[prefix + 'tests'] = {}
    stats['mean_f1'] = metrics_df['f1'].mean()
    stats['highest_f1'] = metrics_df.sort_values('f1', ascending=False).iloc[:5]['f1'].to_dict()
    stats['lowest_f1'] = metrics_df.sort_values('f1', ascending=True).iloc[:5]['f1'].to_dict()
    stats['lower_precision_threshold'] = metrics_df[metrics_df['precision'] < prec_thres]['precision'].sort_values().to_dict()
    stats['lower_recall_threshold'] = metrics_df[metrics_df['recall'] < prec_thres]['recall'].sort_values().to_dict()
    
    print(prefix + 'tests')
    print(f'  Mean of F1 for all classes: {stats["mean_f1"]:.2f}')
    print(f'  Number of classes with precisions lower than {prec_thres}: {len(stats["lower_precision_threshold"])}')
    print(f'  Number of classes with recalls lower than {recall_thres}: {len(stats["lower_recall_threshold"])}')
    
    if gs_job_dir:
        gs_job_dir = gs_job_dir + 'stats/'
        print('  Copy metrics and confusions files to gscloud')
        os.system(f'gsutil cp {metrics_file} {gs_job_dir}')
        os.system(f'gsutil cp {confusion_file} {gs_job_dir}')
    
    # Test fails if either mean f1 is lower than threshold or precision/recall of any class is lower than threshold
    return stats['mean_f1'] >= mean_f1_thres and not stats['lower_precision_threshold'] and not stats['lower_precision_threshold']
    
def evaluate(validation_yaml, unit_test_yaml, handmade_test_yaml, model, 
             gs_job_dir=None, img_size=1280, min_opt_conf=0.6, val_map_thres=0.5,
             unit_mean_f1_thres=0.5, unit_prec_thres=0.5, unit_recall_thres=0.5,
             handmade_mean_f1_thres=0.5, handmade_prec_thres=0.5, handmade_recall_thres=0.5):
    """Evaluate the model against a number of tests.
    Return None if the model fails. Return the confidence threshold if it's sucessful.
    """
    print('\n==================== EVALUATING MODELS ===================\n')
    
    # 1a. Compute validation mAP
    # 1b. Retrieve optimal confidence threshold for following metrics calculation and inference usage
    print(f'EVALUATE {validation_yaml}', flush=True)
    val_map, conf = run(validation_yaml, model, imgsz=img_size, get_optimal_conf=True)
    conf = max(conf, min_opt_conf)
    val_status = val_map >= val_map_thres
    results = { "mean_map_validation": val_map }
    print(f'Optimal confidence: {conf:.2f}')
    print(f'validation mAP: {val_map:.2f}')
    print('val_status', val_status)
    
    # 2. Unit tests
    unit_test_status = run_test(
        unit_test_yaml, model, conf=conf, results=results, gs_job_dir=gs_job_dir, prefix='unit_', 
        mean_f1_thres=unit_mean_f1_thres, prec_thres=unit_prec_thres, recall_thres=unit_recall_thres, img_size=img_size)
    print('unit_test_status', unit_test_status)
    
    # 3. Handmade tests
    handmade_test_status = run_test(
        handmade_test_yaml, model, conf=conf, results=results, gs_job_dir=gs_job_dir, prefix='handmade_',
        mean_f1_thres=handmade_mean_f1_thres, prec_thres=handmade_prec_thres, recall_thres=handmade_recall_thres, img_size=img_size)
    print('handmade_test_status', handmade_test_status)

    # Write results
    save_folder = Path('__temp__')
    save_folder.mkdir(exist_ok=True)
    results_file = save_folder/'results.json'
    with open(results_file, 'w') as f:
        f.write(json.dumps(results))
        
    if gs_job_dir:
        print('Copy results file to gscloud')
        os.system(f'gsutil cp {results_file} {gs_job_dir}')
        
    if val_status and unit_test_status and handmade_test_status:
        print('All tests suceeded')
        return conf
    else:
        print('Tests failed; see results.json for more details')
        return None
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', type=str, required=True, help='GCS location for writing checkpoints and exporting models')
    parser.add_argument('--model-pointers-dir', type=str, required=True, help='GCS location that keeps pointers to job dirs')
    parser.add_argument('--circle-ci-token', type=str, required=True, help='to deploy in circleci')
    parser.add_argument('--train-folder', type=str, required=True, help='name of the training data folder')
    parser.add_argument('--unittest-folder', type=str, required=True, help='name of the unit test data folder')
    parser.add_argument('--handmade-folder', type=str, default='handmade', help='name of the handmade test data folder')
    parser.add_argument('--min-opt-conf', type=float, default=0.5, help='minimum optimal confidence threshold')
    parser.add_argument('--val-map-thres', type=float, default=0.5, help='minimum mAP of validation set')
    parser.add_argument('--unit-mean-f1-thres', type=float, default=0.5, help='minimum average f1 of unittests')
    parser.add_argument('--unit-prec-thres', type=float, default=0.5, help='minimum precision of all classes of unittests')
    parser.add_argument('--unit-recall-thres', type=float, default=0.5, help='minimum recall of all classes of unittests')
    parser.add_argument('--handmade-mean-f1-thres', type=float, default=0.5, help='minimum average f1 of handmade tests')
    parser.add_argument('--handmade-prec-thres', type=float, default=0.5, help='minimum precision of all classes of handmade tests')
    parser.add_argument('--handmade-recall-thres', type=float, default=0.5, help='minimum recall of all classes of handmade tests')
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--img-size', type=int, default=640, help='image sizes')
    opt = parser.parse_known_args()[0]
    

    # 1. Evaluate
    validation_yaml = 'data/' + opt.train_folder + '/data.yaml'
    unit_test_yaml = 'data/' + opt.unittest_folder + '/data.yaml'
    handmade_test_yaml = 'data/' + opt.handmade_folder + '/data.yaml'
    model = 'runs/train/exp/weights/best.pt'

    optimal_conf = evaluate(validation_yaml, unit_test_yaml, handmade_test_yaml, model, 
                            gs_job_dir=opt.job_dir, 
                            min_opt_conf=opt.min_opt_conf,
                            val_map_thres=opt.val_map_thres,
                            unit_mean_f1_thres=opt.unit_mean_f1_thres, 
                            unit_prec_thres=opt.unit_prec_thres, 
                            unit_recall_thres=opt.unit_recall_thres,
                            handmade_mean_f1_thres=opt.handmade_mean_f1_thres, 
                            handmade_prec_thres=opt.handmade_prec_thres, 
                            handmade_recall_thres=opt.handmade_recall_thres)
    
    # 2. Deploy
    if optimal_conf is not None:
        model_details = {
            'size': opt.img_size,
            'base': opt.weights,
            'epochs': opt.epochs,
            'model': model
        }

        update_metadata(optimal_conf, model_details, opt.job_dir, opt.model_pointers_dir)
        deploy(opt.circle_ci_token)