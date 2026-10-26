import argparse
import zipfile
import os
import sys
import glob
import shutil
#sys.path.append('/home/region-profiler/scripts') # adds 'scripts' to search environment, contains utils.py
#from utils import run_sample

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="input")
parser.add_argument("--output", help="output")
parser.add_argument("--glioma_subtype", help="glioma_subtype")
parser.add_argument("--k_value", help="k_value")
parser.add_argument("--codido", help="running on codido", default=False)
############################################
# TODO: add extra args here
############################################

args = parser.parse_args()

#input_folder_path = os.path.join(os.sep, 'app', 'inputs')
#output_folder_path = os.path.join(os.sep, 'app', 'outputs')
input_folder_path = '/data/havoc_profiler/svs_files/codido'
output_folder_path = '/data/havoc_profiler_study'
# inputs and outputs folders need to be created in Dockerfile to allow being mounted
os.makedirs(input_folder_path, exist_ok=True)
#os.makedirs(output_folder_path, exist_ok=True)
print(args)

if args.codido == 'True':
    import boto3

    s3 = boto3.client('s3')

    # downloads codido input file into the folder specified by input_folder_path
    input_file_path = os.path.join(input_folder_path, args.input.split('_SPLIT_')[-1])
    file_svs = args.input.split('_SPLIT_')[-1]
    s3.download_file(os.environ['S3_BUCKET'], args.input, input_file_path)
else:
    input_file_path = glob.glob(os.path.join(input_folder_path, '*'))[0]
    file_svs = os.listdir(input_folder_path)[0]

#shutil.copy(os.path.join(input_folder_path, args.input.split('_SPLIT_')[-1]),
#            os.path.join('/data', 'havoc_profiler', 'svs_files', 'default_project'))


cmd_inner = ['cd /app/src;', 
             'python3', '/app/src/app.py',
             '--file_svs', file_svs, 
             '--k_value', str(args.k_value),
             '--glioma_subtype', args.glioma_subtype]
cmd = ' '.join(cmd_inner)
print(cmd)
os.system(cmd)


cmd_inner = ['cd /data/havoc_profiler_study;','zip', '-rq', 'codido.zip', 'codido/']
cmd = ' '.join(cmd_inner)
print(cmd)
os.system(cmd)

############################################
# TODO: outputs should be saved to output_folder_path like follows:
#with open(os.path.join(output_folder_path, 'TODO.txt'), 'w') as f:
#    f.write('Hello World')
############################################

if args.codido == 'True':
    s3.upload_file('/data/havoc_profiler_study/codido.zip', os.environ['S3_BUCKET'], args.output)

