"""
Performs classification of all the tiles found in 'hovernet/study/sample_region/tiles_selected'

Usage:
  dlfv_fms.py --vit_model=<NAME> --study=<NAME> --dir_tiles=<DIR> --dir_dlfv=<DIR>
  dlfv_fms.py --vit_model <NAME> --study <NAME> --dir_tiles=<DIR> --dir_dlfv <DIR>

Options:
  --vit_model <NAME>    kaiko_large_patch14
  --study <NAME>        TCGA-GBM__k6_px512_mpp252_kumar
  --dir_tiles <DIR>     /data/havoc-profiler/hovernet/mystudy or /data2/havoc_profiler_study/mystudy/hovernet
  --dir_dlfv <DIR>      /data/havoc-profiler/dlfv or /data2/havoc_profiler_study/mystudy
"""

from docopt import docopt

import os
import csv
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms.v2 as v2 # kaiko
from torchvision import transforms     # mahmoodlab_uni2h, prov_gigapath
import timm
from timm.layers import SwiGLUPacked
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from PIL import Image, UnidentifiedImageError # Import UnidentifiedImageError
import glob
import runpy
from typing import Optional, Any, List, Tuple # Added for robust typing
from datetime import datetime


class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        self.class_to_idx = {cls: i for i, cls in enumerate(os.listdir(root)) if os.path.isdir(os.path.join(root, cls))}
        for cls in self.class_to_idx:
            path_pattern = os.path.join(root, cls, 'tiles_selected', '*.*')
            for fname in glob.glob(path_pattern):
                self.samples.append((fname, self.class_to_idx[cls]))
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Optional[Tuple[Any, int, str]]:
        """
        Loads the image and returns the sample. Returns None if loading fails.
        Also returns the path to ensure correct association in the main loop.
        """
        path, label = self.samples[idx]
        
        try:
            # --- START: ROBUST LOADING FIX ---
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            # Return path as well, this is crucial for the main loop
            return image, label, path
        
        except UnidentifiedImageError as e:
            # Log the error and path, then return None
            print(f"[WARNING] Skipping bad image: '{path}'. Error: {e}")
            return None
        except Exception as e:
            # Catch other potential loading errors
            print(f"[ERROR] Skipping image due to unexpected error: '{path}'. Error: {e}")
            return None
        # --- END: ROBUST LOADING FIX ---

# --- START: MODIFIED COLLATE FUNCTION ---
def custom_collate_fn(batch: List[Optional[Tuple[torch.Tensor, int, str]]]) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Filters out the None values (failed samples) and collates the rest.
    Returns batches of images, labels, and paths.
    Now with error handling for the collation step itself.
    """
    try:
        # 1. Filter out None (bad samples)
        batch = [sample for sample in batch if sample is not None]

        # 2. Check for empty batch (if all samples in batch failed)
        if not batch:
            print("[INFO] Batch was empty after filtering, skipping.")
            # Return empty tensors and an empty list
            return torch.empty(0), torch.empty(0), [] 

        # 3. Unzip the valid samples
        images = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        paths = [item[2] for item in batch] # Pass the paths along

        # 4. Stack tensors - THIS IS A POTENTIAL FAILURE POINT
        images_batched = torch.stack(images)
        labels_batched = torch.tensor(labels)

        # 5. Return all three
        return images_batched, labels_batched, paths
    
    except Exception as e:
        # If collating fails (e.g., images are not the same size), print error and skip batch
        print(f"[ERROR] Failed to collate batch: {e}. Skipping batch.")
        # Try to log the paths that were in the failing batch
        try:
            failed_paths = [sample[2] for sample in batch if sample]
            if failed_paths:
                print(f"Paths in failed batch: {failed_paths}")
        except:
            pass # Don't fail while trying to log a failure
        
        # Return an empty batch to continue processing
        return torch.empty(0), torch.empty(0), []
# --- END: MODIFIED COLLATE FUNCTION ---


if __name__ == "__main__":
    s = datetime.now()
    args = docopt(__doc__)
    # Path to your folder (must contain at least one subfolder for ImageFolder)
    #image_folder = f"/data/havoc_profiler/hovernet/{args['--study']}"
    # Save as CSV
    csv_path = f"{args['--dir_dlfv']}/vit_{args['--study']}_{args['--vit_model']}.csv"
    # Pick device
    # --- TYPO FIX HERE ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Preprocessing
    transform_kaiko = v2.Compose([
        v2.ToImage(),
        v2.Resize(size=224),
        v2.CenterCrop(size=224),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    transform_mahmoodlab_uni2h = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    transform_prov_gigapath = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    model = None
    preprocessing = None
    if args['--vit_model']=='kaiko_large_patch14':
      model = timm.create_model(
          model_name="hf-hub:1aurent/vit_large_patch14_reg4_224.kaiko_ai_towards_large_pathology_fms",
          dynamic_img_size=True,
          pretrained=True)
      preprocessing = transform_kaiko
    if args['--vit_model']=='kaiko_base_patch16':
      model = timm.create_model(
          model_name="hf-hub:1aurent/vit_base_patch16_224.kaiko_ai_towards_large_pathology_fms",
          dynamic_img_size=True,
          pretrained=True)
      preprocessing = transform_kaiko
    if args['--vit_model']=='prov_gigapath':
      model = timm.create_model(
          model_name="hf_hub:prov-gigapath/prov-gigapath",
          dynamic_img_size=True,
          pretrained=True)
      preprocessing = transform_prov_gigapath
    if args['--vit_model']=='virchow2':
      model = model = timm.create_model('hf-hub:paige-ai/Virchow2', 
          pretrained=True, 
          mlp_layer=SwiGLUPacked, 
          act_layer=torch.nn.SiLU)    
      preprocessing = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    if args['--vit_model']=='mahmoodlab_uni2h':
      res = runpy.run_path('/app/src/fm_loader/hf_mahmoodlab_uni2h.py')
      model = res["model"]
      preprocessing = transform_mahmoodlab_uni2h
    
    # Model to GPU
    model.to(device).eval()
    
    
        # Dataset & DataLoader
    dataset = CustomDataset(root=args['--dir_tiles'], transform=preprocessing)
    
    # --- MODIFIED DATALOADER ---
    # Added collate_fn to handle 'None' samples
    #
    # *** DEBUGGING FIX: Set num_workers=0 ***
    # This disables multiprocessing for data loading.
    # It will be slower but will prevent worker crashes and give
    # a proper error message if a crash still occurs.
    dataloader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=0, # <-- CRITICAL CHANGE FOR DEBUGGING
        collate_fn=custom_collate_fn,
        # persistent_workers is not needed when num_workers=0
    )
    # --- END MODIFICATION ---
    
    
    # Collect features & paths
    all_rows = []
    total_skipped = 0
    with torch.no_grad():
        # --- MODIFIED LOOP ---
        # The dataloader now returns (batch, labels, paths)
        # We use _ for labels since they aren't used here
        for batch_idx, (batch, _, paths) in enumerate(dataloader):
            
            # Check if batch was empty after filtering
            if batch.numel() == 0:
                print(f"[INFO] Skipping empty batch {batch_idx}")
                total_skipped += 1 # or add len(batch) from collate_fn if we passed it
                continue
                
            # Move batch to GPU
            batch = batch.to(device)
            # Forward pass
            output = model(batch)  # shape: (batch_size, num_features)
            
            # --- START: FIX FOR 3D TENSOR OUTPUT ---
            # ViT models like Virchow2 often return all token features (B, N, D)
            # We only want the [CLS] token's features, which is at index 0.
            if output.dim() == 3:
                output = output[:, 0, :] # Select features for the [CLS] token
            # --- END: FIX ---
            
            
            # Move back to CPU for saving
            features = output.cpu().tolist()
            
            # Image paths are now passed directly from the collate_fn
            for i, feat_vec in enumerate(features):
                # We now use the path from the 'paths' list, which is guaranteed
                # to be correct even if other samples were skipped.
                img_path = paths[i] 
                #row = [img_path] + feat_vec
                formatted_vec = [f"{val:.4f}" for val in feat_vec]
                row = [img_path] + formatted_vec
                all_rows.append(row)
        # --- END MODIFICATION ---

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Check if all_rows is empty before trying to access all_rows[0]
        if all_rows:
            header = ["image_path"] + [f"f{i}" for i in range(len(all_rows[0]) - 1)]
            writer.writerow(header)
            writer.writerows(all_rows)
        else:
            print("[WARNING] No valid images were processed. CSV file will be empty.")
            writer.writerow(["image_path"]) # Write header even if empty
    
    print(f"✅ Saved features for {len(all_rows)} images to {csv_path}")
    if total_skipped > 0:
        print(f"⚠️ Skipped {total_skipped} batches due to collation errors.")
    e = datetime.now()
    elapsed = (e - s).total_seconds()
    print("Elapsed:", elapsed, "seconds")

