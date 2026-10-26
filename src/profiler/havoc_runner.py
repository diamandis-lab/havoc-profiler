import os
import pandas as pd


class Runner:

    def __init__(self, args):
        self.dir_svs = args.get('--dir_svs', None)
        self.file_svs = args.get('--file_svs', None)
        self.dir_havoc = args.get('--dir_havoc', None)
        if 'run_havoc' in args.keys():  # if supplied or created by docopt
            if args['run_havoc']:       # only if supplied
                self.tile_size = int(args['--tile_size'])
                self.desired_tile_mpp = float(args['--desired_tile_mpp'])
                self.k_val = int(args['--k_val'])

    def run_sample(self):
        from havoc_clustering.havoc import HAVOC
        from havoc_clustering.general_utility.slide import Slide
        s = Slide(
            os.path.join(self.dir_svs, self.file_svs),
            img_requirements={
                'compression': None,  # >=70 preferred, but lower accepted
                'mpp': None  # all mpp values (magnification) are supported currently
            }
        )
        feature_extractor_path = os.path.join('/app', 'model_cnn', '74_class_model.h5')
        havoc = HAVOC(s, feature_extractor_path, self.dir_havoc, 
                      tile_size=self.tile_size, 
                      desired_tile_mpp=self.desired_tile_mpp,
                      safe_mpp=False, 
                      hd_backdrop=False)

        kwargs = {
            'save_thumbnail': True,
            'make_dendrogram': False,
            'make_tsne': False,
            'make_corr_map': False,
            'save_tiles_k_vals': [self.k_val]
        }
        havoc.run(k_vals=[self.k_val], min_non_blank_amt=0.5, layer_name='global_average_pooling2d_1', **kwargs)
        
        # extract image dimensions; needed for plotting the regions
        sample_name = self.file_svs.replace('.svs', '').replace('.ndpi', '')
        fname_out = os.path.join(self.dir_havoc, sample_name, sample_name + '_dimensions.txt')
        with open(fname_out, "w") as file:
            file.write(f"width {s.width}\nheight {s.height}\nmagnification {s.apparent_magnification}\n")


    def find_svs(self):
        os.makedirs(self.dir_havoc, exist_ok=True)
        status_pass = True
        if not os.path.exists(self.dir_havoc):
            print('dir_havoc not found: ' + self.dir_havoc)
            status_pass = False
        if not os.path.exists(self.dir_svs):
            print('dir_svs not found: ' + self.dir_svs)
            status_pass = False
        if status_pass:
            fnames = [fname for fname in os.listdir(self.dir_svs) if '.svs' in fname]
            table = pd.DataFrame({'file': fnames, 'k_value': 3})
            table.to_csv(os.path.join(self.dir_havoc, 'havoc_k.txt'), index=False)
            print('No. svs files found: ' + str(len(fnames)))
            print(
                '\033[91m### Edit the k_value parameter in the file ' + os.path.join(self.dir_havoc, 'havoc_k.txt ###'))

    def validate(self):
        print('dir_svs: ' + self.dir_svs)

        def validate_k(k):
            is_good = False
            if isinstance(k, int):
                if k > 1 and k < 100:
                    is_good = True
            return (is_good)

        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        fname_table = os.path.join(self.dir_havoc, 'havoc_k.txt')
        if os.path.exists(fname_table):
            table = pd.read_csv(fname_table)
            file_found = [os.path.exists(os.path.join(self.dir_svs, fname)) for fname in table['file']]
            table['file_found'] = pd.DataFrame(file_found)
            print(table)
            if sum(file_found) != len(file_found):
                print('\033[91m### Error: svs file(s) missing ###')
            k_pass = [validate_k(k) for k in table['k_value']]
            if sum(k_pass) != len(k_pass):
                print('\033[91m### k_value Error: a k value must be an integer between 2 and 100 ###')

        else:
            print('File not found: ' + fname_table)
