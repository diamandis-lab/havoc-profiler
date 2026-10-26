# workflow4j_local.py: prov_gigapath for DLFV extraction/classification, new glial_histology_v3

import os
import argparse
import subprocess
import zipfile
import glob
import time
from pathlib import Path
import random
import uuid


study_original = 'TCGA-GBM'

k_val = 6
tile_size = 512
desired_tile_mpp=1.008
#desired_tile_mpp=0.252
tile_limit = 50
docker_img = 'havoc-profiler:05d'
gpu_num = "0"
hovernet_model = "kumar"
vit_model = 'prov_gigapath'
#classifier_model = "glial_histology_v3"
classifier_model = None
risk_model = ['risk_v4_IDHmut-codel', 'risk_v4_IDHmut-non-codel', 'risk_v4_IDHwt']
#risk_model = ['risk_v4b_IDHwt']
risk_model = None

def run_docker_cmd(task_name, cmd_inner, use_gpu=True):
  run_id = str(uuid.uuid4())[:8] # unique identifier for containers, avoids duplicate IDs
  cmd_docker = ['docker', 'run', '--rm', f'--gpus \'"device={gpu_num}"\'',
                '--user', '$(id -u):$(id -g)',
                '--name', f'{task_name}_{run_id}',
                '-v', '/data/havoc_profiler:/data/havoc_profiler',
                '-v', '/data/docker/havoc-profiler/src:/app/src',
                '-v', '/data2/havoc_profiler/svs_files:/data/havoc_profiler/svs_files', # /data or /data2
                '--entrypoint', 'bash',
                docker_img]
  if use_gpu==False:
    cmd_docker.remove(f'--gpus \'"device={gpu_num}"\'')
  cmd = ' '.join(cmd_docker) + ' -c \"' + ' '.join(cmd_inner) + '\"'
  print(cmd)
  os.system(cmd)


def launch_havoc(study, sample_fname):
  sample = sample_fname.replace('.svs','').replace('.ndpi', '')
  dir_svs = os.path.join('/data', 'havoc_profiler', 'svs_files', study_original)
  dir_havoc = os.path.join('/data', 'havoc_profiler', 'havoc', study)
  os.makedirs(dir_havoc, exist_ok=True)
  cmd_inner = ['cd /app/src;', 'python3', '/app/src/actions_internal.py', 
                                              "--action", "havoc",
                                              "--study", study,
                                              "--study_original", study_original,
                                              "--file_svs", sample_fname, 
                                              "--tile_size", str(tile_size),
                                              "--desired_tile_mpp", str(desired_tile_mpp),
                                              "--k_val", str(k_val)]
  run_docker_cmd(task_name='havoc', cmd_inner=cmd_inner, use_gpu=True)


def launch_hovernet(study, sample, region, hovernet_model):
  dir_in = os.path.join('/data','havoc_profiler', 'havoc' ,study, sample, 'tiles', str(k_val), region)
  tiles_sel = os.listdir(dir_in)
  dir_out = os.path.join('/data','havoc_profiler', 'hovernet' ,study, sample + '_' + region, 'tiles_selected')
  os.makedirs(dir_out, exist_ok=True)
  if len(tiles_sel) > tile_limit:
    random.seed(123456)
    tiles_sel = random.sample(tiles_sel, tile_limit)
  for tile in tiles_sel:
    cmd = ' '.join(['cp', "'"+os.path.join(dir_in, tile)+"'", "'"+os.path.join(dir_out, tile)+"'"])
    os.system(cmd)
  cmd_inner = ['cd /app/src;', '/app/env_torch/bin/python3', '/app/src/actions_internal.py',
                "--action", "segment", 
                "--hovernet_model", hovernet_model,
                "--dir_hovernet", os.path.join('/data', 'havoc_profiler', 'hovernet', study, sample + '_' + region)]
  #run_docker_cmd(task_name='hovernet', cmd_inner=cmd_inner, use_gpu=True)

def launch_cellprofiler(study):
  # runs all the samples found inside in one run using parallel jobs
  cmd_inner = ['python3', '/app/src/actions_internal.py', '--action', 'analyze', '--study', study, '--threads', '16']
  run_docker_cmd(task_name='cellprofiler', cmd_inner=cmd_inner, use_gpu=False)


def launch_dlfv_extractor(study, vit_model):
  cmd_inner1 = ['/app/env_torch/bin/python3 /app/src/profiler/dlfv_fms.py', '--vit_model', vit_model, '--study', study, '--dir_tiles', f'/data/havoc_profiler/hovernet/{study}', '--dir_dlfv', f'/data/havoc_profiler/dlfv']
  cmd_inner1[len(cmd_inner1)-1] = cmd_inner1[len(cmd_inner1)-1] + ';'
  #cmd_inner1 = ['/app/env_torch/bin/python3 /app/src/profiler/dlfv_fms.py', '--vit_model', vit_model, '--study', study, '--dir_tiles', f'/data/havoc_profiler_study/{study}/hovernet', '--dir_dlfv', f'/data/havoc_profiler/dlfv']
  cmd_inner2 = ['Rscript /app/src/R/amalgamate_dlfv.R', '--dlfv_in', f'/data/havoc_profiler/dlfv/vit_{study}_{vit_model}.csv', '--dir_out', '/data/havoc_profiler/dlfv_region']
  cmd_inner_merge = cmd_inner1 + cmd_inner2
  run_docker_cmd(task_name='dlfv_extractor', cmd_inner=cmd_inner_merge, use_gpu=True)


def launch_region_classifier(study, vit_model, classifier_model):
  cmd_inner = ['Rscript /app/src/R/infer_region_class.R', '--study', study, '--vit_model', vit_model,'--classifier_model', classifier_model, '--threads', '16']
  run_docker_cmd(task_name='region_classifier', cmd_inner=cmd_inner, use_gpu=False)


def launch_risk(study, classifier_model, vit_model, risk_model):
  cmd_inner = ['Rscript /app/src/R/infer_risk.R', 
              '--study', study, 
              '--classifier_model', classifier_model, 
              '--vit_model', vit_model, 
              '--risk_model', risk_model,
              '--threads=4']
  run_docker_cmd(task_name='risk_inference', cmd_inner=cmd_inner, use_gpu=False)


study = study_original + '_k' + str(k_val) + '_px' + str(tile_size) + '_mpp' + str(desired_tile_mpp).split('.')[1] + '_' + hovernet_model

# folders need to exist prior being mounted with docker
for dir in ['havoc', 'hovernet', 'cellprofiler']:
  os.makedirs(os.path.join('/data', 'havoc_profiler', dir, study), exist_ok=True)

for dir in ['dlfv', 'dlfv_region','classifier_region', 'features']:
  os.makedirs(os.path.join('/data', 'havoc_profiler', dir), exist_ok=True)

samples_done = os.listdir(os.path.join('/data', 'havoc_profiler', 'havoc', study))
fnames_all = os.listdir(os.path.join('/data2', 'havoc_profiler', 'svs_files', study_original)) # /data or /data2
fnames_pending = [curr for curr in fnames_all if curr.replace('.svs','').replace('.ndpi', '') not in samples_done]

#fnames_pending = []

for sample_fname in fnames_pending:
  sample = sample_fname.replace('.svs','').replace('.ndpi', '')
  try:
    launch_havoc(study=study, sample_fname=sample_fname)
    regions = os.listdir(os.path.join('/data','havoc_profiler', 'havoc' ,study, sample, 'tiles', str(k_val)))
    for region in regions:
      launch_hovernet(study=study, sample=sample, region=region, hovernet_model=hovernet_model)
    cmd_inner = ['rm', '-rf', os.path.join('/data','havoc_profiler', 'havoc' , study, sample, 'tiles')]
    run_docker_cmd(task_name='clean_tiles', cmd_inner=cmd_inner, use_gpu=False)
  except Exception as e:
    print('##### Error, sample not completed')
    print(e)


launch_dlfv_extractor(study=study, vit_model=vit_model)
'''

launch_cellprofiler(study=study)


if classifier_model!=None:
  #launch_region_classifier(study=study, vit_model=vit_model, classifier_model=classifier_model)
  for risk_model_curr in risk_model:
    launch_risk(study=study, classifier_model=classifier_model, vit_model=vit_model, risk_model=risk_model_curr)


for risk_model_curr in risk_model:
  launch_risk(study=study, classifier_model=classifier_model, vit_model=vit_model, risk_model=risk_model_curr)

'''





