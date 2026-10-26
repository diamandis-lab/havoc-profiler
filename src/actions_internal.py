import os
import argparse
import subprocess
import shutil

import zipfile
import glob


import time
import psutil
from pathlib import Path


def run_havoc(args):
  import profiler.havoc_runner
  args = {'--dir_svs': os.path.join('/data/havoc_profiler','svs_files', args.study_original), 
        '--file_svs': args.file_svs,
        '--dir_havoc': os.path.join('/data/havoc_profiler', 'havoc', args.study),
        '--tile_size': int(args.tile_size),
        '--desired_tile_mpp': float(args.desired_tile_mpp),
        '--k_val': int(args.k_val),
        'run_havoc': True}
  profiler.havoc_runner.Runner(args).run_sample()


def run_segment_old(args):
  from profiler.run_hover_net import run_hover_net
  from profiler.get_mask import generate_mask
  img_formats = ['jpg', 'png', 'tiff']
  
  input_folder_path = os.path.join(os.sep, 'app_data', 'hovernet', 'inputs')
  output_folder_path = os.path.join(os.sep, 'app_data', 'hovernet', 'outputs')
  mask_folder_path = os.path.join(os.sep, 'app_data', 'hovernet', 'outputs', 'mask')
  
  os.makedirs(input_folder_path, exist_ok=True)
  os.makedirs(output_folder_path, exist_ok=True)
  os.makedirs(mask_folder_path, exist_ok=True)
  # Run hover net
  run_hover_net(hovernet_model=args.hovernet_model)
  
  input_files_path = glob.glob(os.path.join(input_folder_path, '*'))
  for file in input_files_path:
      if (Path(file).suffix[1:] in (img_formats)):
          # Generate masks for all image 
          generate_mask(file)
  #shutil.rmtree(os.path.join(output_folder_path,'json'))
  shutil.rmtree(os.path.join(output_folder_path,'mat'))
  

def run_segment(args):
  from profiler.hovernet_runner import run_hover_net
  from profiler.get_mask import generate_mask
  img_formats = ['jpg', 'png', 'tiff']
  mask_folder_path = os.path.join(args.dir_hovernet, 'mask')
  os.makedirs(mask_folder_path, exist_ok=True)
  run_hover_net(hovernet_model=args.hovernet_model, dir_hovernet=args.dir_hovernet)
  input_files_path = glob.glob(os.path.join(args.dir_hovernet, 'tiles_selected', '*'))
  for file in input_files_path:
      if (Path(file).suffix[1:] in (img_formats)):
          generate_mask(img_path=file, dir_hovernet=args.dir_hovernet)
  #shutil.rmtree(os.path.join(output_folder_path,'json'))
  shutil.rmtree(os.path.join(args.dir_hovernet, 'mat'))



def run_analyze(args):
  pipeline_file = 'pipeline_v3.cppipe'
  samples_all = os.listdir(os.path.join('/data', 'havoc_profiler', 'hovernet', args.study))
  samples_pending = []
  for sample in samples_all:
    if(not os.path.isfile(os.path.join('/data', 'havoc_profiler', 'cellprofiler', args.study, sample, 'features', 'Image.csv'))):
      samples_pending.append(sample)
  print('run_analyze: no. samples pending=' + str(len(samples_pending)))
  print(samples_pending)
  for sample in samples_pending:
    dir_sample = os.path.join('/data', 'havoc_profiler', 'hovernet', args.study, sample)
    os.makedirs(os.path.join(dir_sample, 'merge'), exist_ok=True)
    shutil.copytree(os.path.join(dir_sample, 'tiles_selected'), os.path.join(dir_sample, 'merge'), dirs_exist_ok=True)
    fnames_mask = os.listdir(os.path.join(dir_sample, 'mask'))
    for fname_in in fnames_mask:
      fname_out = fname_in.replace('.png', '_mask.png')
      shutil.copy2(os.path.join(dir_sample, 'mask', fname_in), os.path.join(dir_sample, 'merge', fname_out))
  commands = []
  for sample in samples_pending:
    cmd = ['cellprofiler', '-c', '-r', '-p', pipeline_file, 
    '-i', os.path.join('/data', 'havoc_profiler','hovernet', args.study, sample, 'merge'), 
    '-o', os.path.join('/data', 'havoc_profiler', 'cellprofiler', args.study, sample)]
    commands.append(' '.join(cmd))
  with open('/data/havoc_profiler/cellprofiler/jobs_analysis.txt', 'w+') as f:
    for items in commands:
  	  f.write('%s\n' %items)
    f.close()
  os.system('cat /data/havoc_profiler/cellprofiler/jobs_analysis.txt | parallel -j ' + args.threads)
  os.system('rm /data/havoc_profiler/cellprofiler/jobs_analysis.txt')
  for sample in samples_pending:
    dir_sample = os.path.join('/data', 'havoc_profiler', 'hovernet', args.study, sample)
    shutil.rmtree(os.path.join(dir_sample, 'merge'))
  # amalgamates morphometric features, from sample-level files to a single study-level one
  cmd = ['Rscript', os.path.join('/app', 'src', 'R', 'amalgamate.R'),
         '--study', args.study, 
         '--dir_in', os.path.join('/data', 'havoc_profiler', 'cellprofiler', args.study),
         '--dir_out', os.path.join('/data', 'havoc_profiler', 'features'), 
         '--threads', '16']
  print(cmd)
  subprocess.run(cmd)
  



if __name__ == '__main__':
  str_desc = '...'
  str_usage = 'python3 run_segment_analyze.py --action [havoc|segment|analyze] (--threads 8)'
  
  parser = argparse.ArgumentParser(description=str_desc, usage=str_usage)
  parser.add_argument('--action', dest='action', required=True, choices=['havoc', 'havoc2', 'segment', 'analyze', 'launch_havoc', 'launch_hovernet', 'launch_cellprofiler'])
  parser.add_argument('--threads', dest='threads', required=False, default='8')
  parser.add_argument('--file_svs', dest='file_svs', required=False)
  parser.add_argument('--tile_size', dest='tile_size', required=False)
  parser.add_argument('--desired_tile_mpp', dest='desired_tile_mpp', required=False, default='0.504')
  parser.add_argument('--study', dest='study', required=False)
  parser.add_argument('--study_original', dest='study_original', required=False)
  parser.add_argument('--k_val', dest='k_val', required=False)
  parser.add_argument('--hovernet_model', dest='hovernet_model', required=False, choices=['kumar', 'pannuke'])
  parser.add_argument('--dir_hovernet', dest='dir_hovernet', required=False)
  
  os.chdir('/app/src')
  args = parser.parse_args()
  print(args)
  if args.action=='segment':
    run_segment(args)
  if args.action=='analyze':
    run_analyze(args)
  if args.action=='havoc':
    run_havoc(args)




