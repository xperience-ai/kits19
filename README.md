# KITS19 training and inference pipeline
(Note that code and this instruction are in draft state due to rush with the competition, to be refactored)

- setup SparseConvNet framework as described in [https://github.com/facebookresearch/SparseConvNet#setup]
- get train/val/test data for kits19 at [https://github.com/neheller/kits19] and put into corresponding root repo folders
- copy the code from this repo into SparseConvNet/examples/ScanNet/ folder:
    - convert.py - data format convert utils
    - data.py - config parameters for train and inference
    - infer.py - main for the inference code
    - unet.py - main for training code
    - validate.py - main for validation
    - visualize.py - main for exporting png images from the CT scan
