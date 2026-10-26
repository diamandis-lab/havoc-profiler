"""
Takes an svs file as input, using the HAVOC library it extracts as text file the following parameters: width, height and magnification.
If a file name is not supplied, all the svs files present in the input folder will be processed.

Usage:
  havoc_dimensions.py --dir_in=<DIR> --dir_out=<DIR> [--fname=<FNAME>] 

Options:
  --dir_in <DIR>       dir_in
  --dir_out <DIR>      dir_out
  --fname <FNAME> TCGA-AA-AAAA-01-01Z-DX1.svs
"""
import sys
import os
from havoc_clustering.general_utility.slide import Slide
from docopt import docopt
import pprint
from pathlib import Path

def get_dimensions(dir_in, dir_out, fname):
  dir_svs = os.path.join('/data', 'havoc_profiler', 'svs_files')
  s = Slide(
      os.path.join(dir_in, fname)
  )
  fname_out = os.path.join(dir_out, 'dimensions_' + fname.replace('.svs', '.txt').replace('.ndpi', '.txt'))
  print('havoc_dimensions.py:::output_file: ' + fname_out)
  folder_path = Path(dir_out)
  
  folder_path.mkdir(parents=True, exist_ok=True)
  with open(fname_out, "w") as file:
      file.write(f"width {s.width}\nheight {s.height}\nmagnification {s.apparent_magnification}\n")



if __name__ == "__main__":
  args = docopt(__doc__)
  print('starting... havoc_dimensions.py ')
  pprint.pprint(args)
  #get_dimensions(dir_in=args['--dir_in'], dir_out=args['--dir_out'], fname_warc_gz=args['--fname'])
  if args['--fname']==None:
    fnames = os.listdir(args['--dir_in'])
    fnames = [fname for fname in fnames if '.svs' in fname]
    print('no. svs files found=' + str(len(fnames)))
    if len(fnames)==0:
      sys.exit()
  else:
    fname = args['--fname']
    if '.svs' not in fname:
      print('ERROR: the supplied --fname is not an svs file')
      sys.exit()
    if os.path.isfile(os.path.join(args['--dir_in'], args['--fname']))==False:
      print('file does not exist')
      sys.exit()
    fnames = [fname]
  for fname in fnames:
    get_dimensions(dir_in=args['--dir_in'], dir_out=args['--dir_out'], fname=fname)

    
