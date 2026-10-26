# docker run -it --rm -u $(id -u):$(id -g) --gpus=all -v /data:/data --name countic-cnn countic-cnn:01 bash
# python3 python/infer_cnn.py --model_file=models/skcm_v01.h5 --model_layer=global_average_pooling2d_1 --dir_tiles=tiles/TCGA-SKCM/TCGA-3N-A9WD-01Z-00-DX1 --dir_dlfv=DLFV/TCGA-SKCM


"""
Usage:
  infer_cnn.py --model_file=<FILE> --model_layer=<NAME> --dir_tiles=<DIR> --dir_dlfv=<DIR>

Options:
  --model_file <FILE>   /path/to/vgg_model.h5
  --model_layer <NAME>  global_average_pooling2d_1
  --dir_tiles <DIR>     img/img_TCGA-SKCM/TCGA-D3-A8GV-06Z-00-DX1_files/0 (must include sample name)
  --dir_dlfv <DIR>      DLFV/TCGA-SKCM

"""
from docopt import docopt
import pprint


from keras.models import load_model
import numpy as np
import pandas as pd
import keras.backend as K
import cv2
import os
import subprocess
import shutil
import tensorflow as tf
from datetime import datetime


def infer_features(model, layer_name, image_path):
  layer = model.get_layer(name=layer_name)
  image = cv2.imread(image_path)
  #pct_blank = np.sum(np.std(image, axis=2) < 2.75) / (image.shape[0] * image.shape[1])
  pct_blank = 0 # filter disabled for speed test
  if pct_blank<0.5:
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.grayscale_to_rgb(image)
    img = np.expand_dims(image, axis=0) / 255
    input = model.layers[0].input
    output = layer.output
    get_output = K.function([input], [output])
    features = get_output([img])[0].flatten()
  else:
    #print('pct_blank=' + str(pct_blank) + ' ' + image_path)
    features = None
  return(features)

def execCMD(cmd):
  cmd = ' '.join(cmd)
  print(cmd)
  output = subprocess.check_output(cmd, shell=True)
  output = output.decode("utf-8")
  output = output.split('\n')
  print(output)
  return(output)

def inferDLFV(model_file, model_layer, dir_tiles, dir_dlfv):
  model_name = os.path.basename(model_file).replace('.h5','')
  sample_name = os.path.basename(dir_tiles.replace('_files/0',''))
  
  model = load_model(model_file)
  #model.summary()
  
  tiles = os.listdir(dir_tiles)
  #tiles = tiles[:1000]
  features = np.empty([0, 0])
  print('Model: ' + model_name + ' Sample: ' + sample_name + ' processing ' + str(len(tiles)) + ' tiles')
  pos = 0
  tiles_failed = []
  for fname in tiles:
    pos = pos + 1
    if pos%100==0:
      print(str(pos) + '/' + str(len(tiles)))
    try:
      features_curr = infer_features(model, layer_name=model_layer, image_path=os.path.join(dir_tiles, fname))
      # pct_blank > 0.5 were skipped, assigned 'None'
      if features_curr is not None:
        features_curr = features_curr.reshape((np.size(features_curr), 1))
        if features.size == 0:
          features = features_curr
        else:
          features = np.hstack((features, features_curr))
      else:
        tiles_failed.append(fname) # skipped blank tile
    except:
      tiles_failed.append(fname)
      print('###### ERROR, file skipped: ' + str(pos) + ' ' + fname)
  DLFV = pd.DataFrame(features)
  tiles_done = [x for x in tiles if x not in tiles_failed]
  DLFV.columns = [tile.split('.')[0] for tile in tiles_done]
  feature_names = [model_name + '|' +str(x) for x in DLFV.index]
  DLFV = DLFV.transpose()
  DLFV.columns = feature_names
  os.makedirs(dir_dlfv, exist_ok=True)
  fname_dlfv = dir_dlfv + '/DLFV_' + sample_name + '_' + model_name + '.csv.gz'
  DLFV.to_csv(fname_dlfv, compression="gzip")
  print('File saved: ' + fname_dlfv)



if __name__ == "__main__":
  s = datetime.now()
  args = docopt(__doc__)
  pprint.pprint(args)
  dir_base = '/home'
  os.chdir(dir_base)
  
  
  
  inferDLFV(model_file=args['--model_file'],
            model_layer=args['--model_layer'],
            dir_tiles=args['--dir_tiles'], 
            dir_dlfv=args['--dir_dlfv'])
  e = datetime.now()
  elapsed = (e - s).total_seconds()
  print("Elapsed:", elapsed, "seconds")

