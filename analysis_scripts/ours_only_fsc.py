import os
from datetime import datetime as dt
import logging
import numpy as np
import torch
import argparse
import pickle
import glob, re
from cryodrgn.commands_utils.fsc import calculate_fsc
from cryodrgn.source import ImageSource

log = print
logger = logging.getLogger(__name__)

def add_args(parser):
    parser.add_argument('input_dir', help='directory that contains volumes')
    parser.add_argument('-o', help='Output directory')
    parser.add_argument('--gt-dir', help='Directory of gt volumes')
    parser.add_argument("--mask", default=None, help="Use mask to compute the masked metric")
    parser.add_argument("--invert", action="store_true", help="Invert contrast of output volume")
    parser.add_argument('--overwrite',action='store_true')
    parser.add_argument('--cuda-device', default=0, type=int)
    # gen vols
    parser.add_argument(
        "--vol-start-index",
        type=int,
        default=0,
        help="Default value of start index for volume generation (default: %(default)s)",
    )
    parser.add_argument(
        "--prefix",
        default="vol_",
        help="Prefix when writing out multiple .mrc files (default: %(default)s)",
    )
    return parser

def load_pkl(pkl: str):
    with open(pkl, "rb") as f:
        x = pickle.load(f)
    return x

def get_cutoff(fsc, t):
    w = np.where(fsc[:,1] < t)
    log(w)
    if len(w[0]) >= 1:
        x = fsc[:,0][w]
        return 1/x[0]
    else:
        return 2

def natural_sort_key(s):
    # Convert the string to a list of text and numbers
    parts = re.split('([0-9]+)', s)
    
    # Convert numeric parts to integers for proper numeric comparison
    parts[1::2] = map(int, parts[1::2])
    
    return parts

def main(args):
    t1 = dt.now()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    file_pattern = "*.mrc"
    files = glob.glob(os.path.join(args.gt_dir, file_pattern))
    gt_dir = sorted(files, key=natural_sort_key)
    
    # Compute FSC cdrgn
    if not os.path.exists('{}/fsc'.format(args.o)):
        os.makedirs('{}/fsc'.format(args.o))
    if not os.path.exists('{}/fsc_no_mask'.format(args.o)):
        os.makedirs('{}/fsc_no_mask'.format(args.o))
    
    for ii in range(len(gt_dir)):
        print('gt_dir[ii]:',gt_dir[ii])
        if args.mask is not None:
            out_fsc = '{}/fsc/{}.txt'.format(args.o, ii)
        else:
            out_fsc = '{}/fsc_no_mask/{}.txt'.format(args.o, ii)

        vol_file = '{}/vol_{:03d}.mrc'.format(args.input_dir, ii)
        print('vol_file:',vol_file)
        vol1 = ImageSource.from_file(gt_dir[ii])
        vol2 = ImageSource.from_file(vol_file)
        if args.invert:
            vol2 *= -1
        # if os.path.exists(out_fsc) and not args.overwrite:
        #     log('FSC exists, skipping...')
        # else:
        fsc_vals = calculate_fsc(vol1.images(), vol2.images(), args.mask)
        np.savetxt(out_fsc, fsc_vals)

    # Summary statistics
    fsc = [np.loadtxt(x) for x in glob.glob('{}/fsc/*txt'.format(args.o))]
    fsc143 = [get_cutoff(x,0.143) for x in fsc]
    fsc5 = [get_cutoff(x,.5) for x in fsc]
    log('cryoDRGN FSC=0.143')
    log('Mean: {}'.format(np.mean(fsc143)))
    log('Median: {}'.format(np.median(fsc143)))
    log('cryoDRGN FSC=0.5')
    log('Mean: {}'.format(np.mean(fsc5)))
    log('Median: {}'.format(np.median(fsc5)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())