"""
Performs classification of all the tiles found in 'hovernet/study/sample_region/tiles_selected'

Usage:
  classifier.py --study=<NAME> --model_name=<NAME>
  classifier.py --study <NAME> --model_name <NAME> 

Options:
  --study <NAME>        TCGA-LGG_k6_px512_pmm252_kumar
  --model_name <NAME>   glial_histology
"""
from docopt import docopt

from keras.models import load_model
import numpy as np
import pandas as pd
import cv2
import keras.backend as K
import os
import tensorflow as tf
import json


class tile_classifier:
  def __init__(self, study, model_name):
    self.dir_hovernet = os.path.join('/app_data', 'hovernet', 'outputs')
    self.dir_classifier = os.path.join('/app_data', 'classifier')
    self.study=study
    self.model_name=model_name
    fname_model = 'fname_model_not_set'
    if os.path.isfile(os.path.join('/app', 'model_cnn', self.model_name, 'model', 'saved_model.pb')):
      # keras model was saved as tensorflow .pb file; requires path to folder containing the .pb and other components
      fname_model = os.path.join('/app', 'model_cnn', self.model_name, 'model')
    if os.path.isfile(os.path.join('/app', 'model_cnn', self.model_name, 'model.keras')):
      # saved as keras model in a zipped '.keras' file
      fname_model = os.path.join('/app', 'model_cnn', self.model_name, 'model.keras')
    self.model = load_model(fname_model)
    #model.summary()
    self.layer_name=self.model.layers[-1].name
    with open('/app/model_cnn/' + self.model_name + '/inference.json') as f:
      model_info = json.load(f)
    self.model_classes = model_info['classes']
    mpp_tile = [int(curr.replace('mpp',''))/1000 for curr in self.study.split('_') if 'mpp' in curr]
    if len(mpp_tile)==1:
      mpp_model = model_info['mpp']
      self.scaling_factor = mpp_tile[0] / mpp_model
      print('mpp_tile=' + str(mpp_tile[0]) + ' mpp_model=' + str(mpp_model) + ' scaling_factor=' + str(self.scaling_factor) )
    else:
      self.scaling_factor = 1
      print('mpp_tile unknown, scaling_factor=1')

    

  def infer_features(self, image_path):
    layer = self.model.get_layer(name=self.layer_name)
    image = cv2.imread(image_path)
    if self.scaling_factor!=1.0:
      image = cv2.resize(image, None, fx=self.scaling_factor, fy=self.scaling_factor)
    img = np.expand_dims(image, axis=0) / 255
    input = self.model.layers[0].input
    output = layer.output
    get_output = K.function([input], [output])
    features = get_output([img])[0].flatten()
    return(features)


  def inferDLFV(self, sample):
    dir_in = os.path.join(self.dir_hovernet, sample, 'tiles_selected')
    dir_out = self.dir_classifier
    tiles = os.listdir(dir_in)
    features = np.empty([0, 0])
    print('Model: ' + self.model_name + ' Sample: ' + sample + ' processing ' + str(len(tiles)) + ' tiles')
    pos = 0
    tiles_failed = []
    for fname in tiles:
      pos = pos + 1
      if pos%100==0:
        print(str(pos) + '/' + str(len(tiles)))
      try:
        features_curr = self.infer_features(image_path=os.path.join(dir_in, fname))
        features_curr = features_curr.reshape((np.size(features_curr), 1))
        if features.size == 0:
            features = features_curr
        else:
            features = np.hstack((features, features_curr))
      except Exception as e:
        tiles_failed.append(fname)
        print('###### ERROR, file skipped: ' + str(pos) + ' ' + fname)
        print(e)
    DLFV = pd.DataFrame(features)
    tiles_done = [x for x in tiles if x not in tiles_failed]
    DLFV.columns = [tile.split('.')[0] for tile in tiles_done]
    feature_names = [self.model_name + '|' +str(x) for x in DLFV.index]
    DLFV = DLFV.transpose()
    DLFV.columns = self.model_classes
    out_file = os.path.join(dir_out, 'infer_' + sample + '_' + self.model_name + '.csv')
    DLFV.to_csv(out_file)
    print('File saved: ' + out_file)


if __name__ == "__main__":
  args = docopt(__doc__)
  tc = tile_classifier(study=args['--study'], model_name=args['--model_name'])
  print('starting... classifier.py ')
  os.makedirs(tc.dir_classifier, exist_ok=True)
  fnames_done = os.listdir(tc.dir_classifier)
  samples_done = [str.replace('infer_', '').replace('_' + tc.model_name + '.csv','') for str in fnames_done]
  dir_hovernet = os.path.join('/data', 'havoc_profiler', 'hovernet', tc.study)
  samples = os.listdir(tc.dir_hovernet)
  samples_pending = [sample for sample in samples if sample not in samples_done]
  for sample in samples_pending:
    try:
      tc.inferDLFV(sample)
    except Exception as e:
      print('##### Error, sample not completed')
      print(e)

  

# docker run -it --rm --gpus "device=0" -v /data:/data -v /data/docker/havoc-profiler:/app havoc-profiler:02 bash -c "python3 /app/src/profiler/classifier.py"


