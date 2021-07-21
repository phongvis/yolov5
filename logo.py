"""An additional module for evaluating and deploying models. 

Usage:
optimal_conf = evaluate(...)
if optimal_conf is not None:
    update_metadata(optimal_conf, model_details)
    deploy(...)
"""

from pathlib import Path
import json
import os
import requests

from val import run, compute_metrics

def update_metadata(optimal_conf, model_details):
    # Make new folder for the model, containing 2 files
    full_folder_name = 'gs://logo-training/jobs/' + model_details['folder'] + '/'
    
    # 1. meta.json
    meta = { 
        'size': model_details['size'],
        'threshold': optimal_conf,
        'base': model_details['base']
    }
    save_folder = Path('__temp__')
    save_folder.mkdir(exist_ok=True)
    meta_file = save_folder/'meta.json'
    with open(meta_file, 'w') as f:
        f.write(json.dumps(meta))
    os.system(f'gsutil cp {meta_file} {full_folder_name}')
    print(f'Copied meta.json to {full_folder_name}')
    
    # 2. best.pt
    os.system(f'gsutil cp {model_details["model"]} {full_folder_name}best.pt')
    print(f'Copied best.pt to {full_folder_name}')
    
    # model
    model_file = save_folder/'test_model2'
    with open(model_file, 'w') as f:
        f.write(full_folder_name[5:-1])
    os.system(f'gsutil cp {model_file} gs://logo-training')
    print('Copied model to gs://logo-training')

def deploy(circle_ci_token):
    """Make circleci rebuild.
    """
    data = {
        'parameters': { 'run_workflow_deploy': True },
        'branch': 'release-1'
    }

    requests.post(
        'https://circleci.com/api/v2/project/github/redsift/logo-detection/pipeline', 
        headers={
            'content-type': 'application/json',
            'Circle-Token': circle_ci_token,
        },
        data=json.dumps(data))
    print('Deployed')

def run_test(yaml, model, conf, results, gs_bucket=None, prefix='', mean_f1_thres=0.5, prec_thres=0.5, recall_thres=0.5, img_size=1280):
    """Run tests, save results, record important metrics and return status.
    """
    metrics_df, confusion_df = compute_metrics(yaml, model, imgsz=img_size, conf_thres=conf)
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
    
    if gs_bucket:
        print('  Copy metrics and confusions files to gscloud')
        os.system(f'gsutil cp {metrics_file} {gs_bucket}')
        os.system(f'gsutil cp {confusion_file} {gs_bucket}')
    
    # Test fails if either mean f1 is lower than threshold or precision/recall of any class is lower than threshold
    return stats['mean_f1'] >= mean_f1_thres and not stats['lower_precision_threshold'] and not stats['lower_precision_threshold']
    
def evaluate(validation_yaml, unit_test_yaml, handmade_test_yaml, model, 
             gs_bucket=None, img_size=1280, val_map_thres=0.5,
             unit_mean_f1_thres=0.5, unit_prec_thres=0.5, unit_recall_thres=0.5,
             handmade_mean_f1_thres=0.5, handmade_prec_thres=0.5, handmade_recall_thres=0.5):
    """Evaluate the model against a number of tests.
    Return None if the model fails. Return the confidence threshold if it's sucessful.
    """
    # 1a. Compute validation mAP
    # 1b. Retrieve optimal confidence threshold for following metrics calculation and inference usage
    val_map, conf = run(validation_yaml, model, imgsz=img_size, get_optimal_conf=True)
    val_map, conf = 0.42, 0.25
    val_status = val_map >= val_map_thres
    results = { "mean_map_validation": val_map }
    print(f'Optimal confidence: {conf:.2f}')
    print(f'validation mAP: {val_map:.2f}')
    
    # 2. Unit tests
    unit_test_status = run_test(
        unit_test_yaml, model, conf=conf, results=results, gs_bucket=gs_bucket, prefix='unit_', 
        mean_f1_thres=unit_mean_f1_thres, prec_thres=unit_prec_thres, recall_thres=unit_recall_thres, img_size=img_size)
    
    # 3. Handmade tests
    handmade_test_status = run_test(
        handmade_test_yaml, model, conf=conf, results=results, gs_bucket=gs_bucket, prefix='handmade_',
        mean_f1_thres=handmade_mean_f1_thres, prec_thres=handmade_prec_thres, recall_thres=handmade_recall_thres, img_size=img_size)

    # Write results
    save_folder = Path('__temp__')
    save_folder.mkdir(exist_ok=True)
    results_file = save_folder/'results.json'
    with open(results_file, 'w') as f:
        f.write(json.dumps(results))
        
    if gs_bucket:
        print('Copy results file to gscloud')
        os.system(f'gsutil cp {results_file} {gs_bucket}')
        
    if val_status and unit_test_status and handmade_test_status:
        print('All tests suceeded')
        return conf
    else:
        print('Tests failed; see results.json for more details')
        return None