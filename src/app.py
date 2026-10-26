"""
Usage:
  app.py --file_svs=<FILE> --k_value=<K_VAL> --glioma_subtype=<NAME>
  

Options:
  --file_svs=<FILE>    Image file
  --glioma_subtype=<NAME> options: unknown,IDHmut-codel,IDHmut-non-codel,IDHwt
  --k_value=<K_VAL>      k_value for Havoc analysis
"""

# docker run -it --gpus=all --rm -u $(id -u):$(id -g) -v /data/havoc_profiler:/data/havoc_profiler --entrypoint bash havoc-profiler-main:01

# python3 /app/src/app.py --file_svs=TCGA-06-0143-01Z-00-DX1.svs --k_value=6 --glioma_subtype=IDHwt

import os
from docopt import docopt
import pprint
import random
from datetime import datetime


#if __name__ == "__main__":
args = docopt(__doc__)
pprint.pprint(args)
k_val = args['--k_value']
tile_size = 512
desired_tile_mpp=0.252
tile_limit = 100
hovernet_model = "kumar"
vit_model = 'prov_gigapath'
classifier_model = "glial_histology_v3"
#risk_model = ['risk_v4_IDHmut-codel', 'risk_v4_IDHmut-non-codel', 'risk_v4_IDHwt']
glioma_subtype = args['--glioma_subtype']
#risk_model = args'risk_v4b_IDHwt'
study_original = 'codido'
#study = study_original + '_k' + str(k_val) + '_px' + str(tile_size) + '_mpp' + str(desired_tile_mpp).split('.')[1] + '_' + hovernet_model
#run_id = study_original + '-' + datetime.now().strftime("%Y%m%d_%H%M")
run_id = study_original + '-xxxxxxxxxx'
study = study_original

risk_cutoff ={'IDHmut-codel':0.327429,
              'IDHmut-non-codel':0.6314468,
              'IDHwt':0.6909994}

############## Havoc #######################
sample = args['--file_svs'].replace('.svs','').replace('.ndpi', '')
dir_svs = '/data/havoc_profiler/havoc/' + study_original
dir_havoc = '/data/havoc_profiler/havoc/' + study
os.makedirs(dir_havoc, exist_ok=True)

cmd_inner = ['cd /app/src;', 'python3', '/app/src/actions_internal.py', 
                                      "--action", "havoc",
                                      "--study", study,
                                      "--study_original", study_original,
                                      "--file_svs", args['--file_svs'], 
                                      "--tile_size", str(tile_size),
                                      "--desired_tile_mpp", str(desired_tile_mpp),
                                      "--k_val", str(k_val)]
print(' '.join(cmd_inner))
os.system(' '.join(cmd_inner))
################ Hovernet ###############
regions = os.listdir(os.path.join(dir_havoc, sample, 'tiles', str(k_val)))

for region in regions:
    dir_in = os.path.join(dir_havoc, sample, 'tiles', str(k_val), region)
    tiles_sel = os.listdir(dir_in)
    dir_out = os.path.join('/data', 'havoc_profiler', 'hovernet' ,study, sample + '_' + region, 'tiles_selected')
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
                "--dir_hovernet", os.path.join('/data/havoc_profiler', 'hovernet', study, sample + '_' + region)]
    print(' '.join(cmd_inner))
    os.system(' '.join(cmd_inner))

################## DLFV extractor ##############
cmd_inner1 = ['/app/env_torch/bin/python3 /app/src/profiler/dlfv_fms.py', '--vit_model', vit_model, '--study', study, '--dir_tiles', f'/data/havoc_profiler/hovernet/{study}', '--dir_dlfv', f'/data/havoc_profiler/dlfv']
os.system(' '.join(cmd_inner1))
cmd_inner2 = ['Rscript /app/src/R/amalgamate_dlfv.R', '--dlfv_in', f'/data/havoc_profiler/dlfv/vit_{study}_{vit_model}.csv', '--dir_out', '/data/havoc_profiler/dlfv_region']
os.system(' '.join(cmd_inner2))
################## Histology classifier ########
cmd_inner = ['Rscript /app/src/R/infer_region_class.R', '--study', study, '--vit_model', vit_model,'--classifier_model', classifier_model, '--threads', '16']
os.system(' '.join(cmd_inner))

################## Risk prediction #############
if glioma_subtype != 'unknown':
  cmd_inner = ['Rscript /app/src/R/infer_risk.R', 
              '--sample_name', sample,
              '--glioma_subtype', glioma_subtype,
              '--classifier_model', classifier_model, 
              '--vit_model', vit_model, 
              '--risk_model', 'xgb_risk_lgggbm_' + glioma_subtype + '_v4',
              '--risk_cutoff', str(risk_cutoff[glioma_subtype]),
              '--threads=4']
  os.system(' '.join(cmd_inner))
###################Cell Profiler ###############
cmd_inner = ['python3', '/app/src/actions_internal.py', '--action', 'analyze', '--study', study, '--threads', '16']
os.system(' '.join(cmd_inner))
################### Package output files ######
dir_out = os.path.join('/data', 'havoc_profiler_study', study, sample)
os.system(f'mkdir -p {dir_out}')

cmd_inner = ['mv', f'/data/havoc_profiler/havoc/{study}/{sample}/*.jpg', dir_out]
#os.system(' '.join(cmd_inner))
cmd_inner = ['mv', f'/data/havoc_profiler/hovernet/{study}', 
                   os.path.join(dir_out, 'hovernet')]
os.system(' '.join(cmd_inner))
cmd_inner = ['cp', f'/data/havoc_profiler/classifier_region/class_region_codido_{classifier_model}.csv', 
                   f'{dir_out}/class_region_{classifier_model}.csv']
os.system(' '.join(cmd_inner))
cmd_inner = ['cp', f'/data/havoc_profiler/risk/{study}/{sample}/*.jpeg', 
                   dir_out]
os.system(' '.join(cmd_inner))
cmd_inner = ['cp', f'/data/havoc_profiler/risk/{study}/{sample}/*.pdf', 
                   dir_out]
os.system(' '.join(cmd_inner))







