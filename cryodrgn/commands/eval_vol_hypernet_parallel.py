import argparse
import os
import pprint
import pickle
import sys
import logging
from datetime import datetime as dt
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import itertools

try:
    import apex.amp as amp  # type: ignore  # PYR01
except ImportError:
    pass

import yaml
import lightning as L

import cryodrgn
from cryodrgn import __version__, ctf
from cryodrgn import dataset
from cryodrgn.lattice import Lattice
from cryodrgn.pose import PoseTracker
from cryodrgn import config
from cryodrgn.mrc import MRCFile

from .hypernetwork_parallel import Hypernet
import cryodrgn.commands.utils

logger = logging.getLogger(__name__)


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "particles",
        type=os.path.abspath,
        help="Input particles (.mrcs, .star, .cs, or .txt)",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=os.path.abspath,
        required=True,
        help="Output directory to save model",
    )
    parser.add_argument(
        "--zdim", type=int, required=True, help="Dimension of latent variable"
    )
    parser.add_argument(
        "--poses", type=os.path.abspath, required=True, help="Image poses (.pkl)"
    )
    parser.add_argument(
        "--ctf", metavar="pkl", type=os.path.abspath, help="CTF parameters (.pkl)"
    )
    parser.add_argument(
        "--load", metavar="WEIGHTS.PKL", help="Initialize training from a checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=1,
        help="Checkpointing interval in N_EPOCHS (default: %(default)s)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1000,
        help="Logging interval in N_IMGS (default: %(default)s)",
    )
    parser.add_argument(
        "--img-list",
        nargs="*",
        type=int,
        help="List of images per conformation or structure",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Increase verbosity"
    )
    parser.add_argument(
        "--seed", type=int, default=np.random.randint(0, 100000), help="Random seed"
    )
    parser.add_argument(
        "-c",
        "--config",
        metavar="YAML",
        required=True,
        help="CryoDRGN config.yaml file",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=1000,
        help="Only create every (skip)th volume",
    )
    parser.add_argument(
        "--load-poses",
        action="store_true",
        help="Load poses from the checkpoint instead of from --poses",
    )

    group = parser.add_argument_group("Dataset loading")
    group.add_argument(
        "--ind",
        type=os.path.abspath,
        metavar="PKL",
        help="Filter particles by these indices",
    )
    group.add_argument(
        "--uninvert-data",
        dest="invert_data",
        action="store_false",
        help="Do not invert data sign",
    )
    group.add_argument(
        "--no-window",
        dest="window",
        action="store_false",
        help="Turn off real space windowing of dataset",
    )
    group.add_argument(
        "--window-r",
        type=float,
        default=0.85,
        help="Windowing radius (default: %(default)s)",
    )
    group.add_argument(
        "--datadir",
        type=os.path.abspath,
        help="Path prefix to particle stack if loading relative paths from a .star or .cs file",
    )
    group.add_argument(
        "--lazy",
        action="store_true",
        help="Lazy loading if full dataset is too large to fit in memory",
    )
    group.add_argument(
        "--shuffler-size",
        type=int,
        default=0,
        help="If non-zero, will use a data shuffler for faster lazy data loading.",
    )
    group.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of subprocesses to use as DataLoader workers. If 0, then use the main process for data loading. (default: %(default)s)",
    )
    group.add_argument(
        "--max-threads",
        type=int,
        default=16,
        help="Maximum number of CPU cores for data loading (default: %(default)s)",
    )
    
    group = parser.add_argument_group("Eval volume parameters")
    group.add_argument(
        "--prefix",
        default="vol_",
        help="Prefix when writing out multiple .mrc files (default: %(default)s)",
    )
    group.add_argument(
        "--flip", action="store_true", help="Flip handedness of output volume"
    )
    group.add_argument(
        "--invert", action="store_true", help="Invert contrast of output volume"
    )
    group.add_argument(
        "-d",
        "--downsample",
        type=int,
        help="Downsample volumes to this box size (pixels)",
    )

    group = parser.add_argument_group("Tilt series parameters")
    group.add_argument(
        "--ntilts",
        type=int,
        default=10,
        help="Number of tilts to encode (default: %(default)s)",
    )
    group.add_argument(
        "--random-tilts",
        action="store_true",
        help="Randomize ordering of tilts series to encoder",
    )
    group.add_argument(
        "--t-emb-dim",
        type=int,
        default=64,
        help="Intermediate embedding dimension (default: %(default)s)",
    )
    group.add_argument(
        "--tlayers",
        type=int,
        default=3,
        help="Number of hidden layers (default: %(default)s)",
    )
    group.add_argument(
        "--tdim",
        type=int,
        default=1024,
        help="Number of nodes in hidden layers (default: %(default)s)",
    )
    group.add_argument(
        "--dose-per-tilt",
        type=float,
        help="Expected dose per tilt (electrons/A^2 per tilt) (default: %(default)s)",
    )
    group.add_argument(
        "-a",
        "--angle-per-tilt",
        type=float,
        default=3,
        help="Tilt angle increment per tilt in degrees (default: %(default)s)",
    )
    
    group = parser.add_argument_group("Training parameters")
    group.add_argument(
        "--norm",
        type=float,
        nargs=2,
        default=None,
        help="Data normalization as shift, 1/scale (default: mean, std of dataset)",
    )

    group = parser.add_argument_group("Encoder Network")
    group.add_argument(
        "--encode-mode",
        default="resid",
        choices=("conv", "resid", "mlp", "tilt"),
        help="Type of encoder network (default: %(default)s)",
    )
    group.add_argument(
        "--enc-mask",
        type=int,
        help="Circular mask of image for encoder (default: D/2; -1 for no mask)",
    )

    group = parser.add_argument_group("From TransINR/PONP")
    group.add_argument(
        "--cfg",
    )
    group.add_argument(
        "--load-root",
        default='data',
    )
    group.add_argument(
        "--save-root",
        default='save',
    )
    group.add_argument(
        '--name', 
        default=None,
    )
    group.add_argument(
        '--tag', 
        default=None,
    )
    group.add_argument(
        '--wandb-upload', 
        '-w', 
        action='store_true',
    )
    group.add_argument(
        '--wandb-yaml', 
        type=str, 
        default='wandb.yaml',
    )
    return parser

def make_cfg(args):
    with open(args.cfg, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    def translate_cfg_(d):
        for k, v in d.items():
            if isinstance(v, dict):
                translate_cfg_(v)
            elif isinstance(v, str):
                d[k] = v.replace('$load_root$', args.load_root)
    translate_cfg_(cfg)

    if args.name is None:
        exp_name = os.path.basename(args.cfg).split('.')[0]
    else:
        exp_name = args.name
    if args.tag is not None:
        exp_name += '_' + args.tag

    env = dict()
    env['exp_name'] = exp_name
    env['save_dir'] = os.path.join(args.save_root, exp_name)
    env['tot_gpus'] = torch.cuda.device_count()
    env['wandb_upload'] = args.wandb_upload
    cfg['env'] = env

    return cfg

def main(args):
    # make cfg
    ti_cfg = make_cfg(args)
    
    logger.info(args)
    
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    cfg = config.load(args.config)
    args.norm = cfg['dataset_args']['norm']

    # load index filter
    if args.ind is not None:
        logger.info("Filtering image dataset with {}".format(args.ind))
        if args.encode_mode == "tilt":
            particle_ind = pickle.load(open(args.ind, "rb"))
            pt, tp = dataset.TiltSeriesData.parse_particle_tilt(args.particles)
            ind = dataset.TiltSeriesData.particles_to_tilts(pt, particle_ind)
        else:
            ind = pickle.load(open(args.ind, "rb"))
    else:
        ind = None
    
    # load dataset
    if args.encode_mode != "tilt":
        args.use_real = args.encode_mode == "conv"  # Must be False
        transform = None
        if cfg.get('transform', False):
            transform = v2.Compose([
                v2.ToTensor(),
                v2.GaussianNoise(sigma=10),
            ])
        data = dataset.ImageDataset(
            mrcfile=args.particles,
            lazy=args.lazy,
            norm=args.norm,
            invert_data=args.invert_data,
            ind=ind,
            keepreal=args.use_real,
            window=args.window,
            datadir=args.datadir,
            window_r=args.window_r,
            max_threads=args.max_threads,
            multi=ti_cfg.get('multi', None),
            shot=ti_cfg.get('shot', 1),
            transform=transform,
        )
    else:
        assert args.encode_mode == "tilt"
        data = dataset.TiltSeriesData(  # FIXME: maybe combine with above?
            args.particles,
            args.ntilts,
            args.random_tilts,
            norm=args.norm,
            invert_data=args.invert_data,
            ind=ind,
            keepreal=args.use_real,
            window=args.window,
            datadir=args.datadir,
            max_threads=args.max_threads,
            window_r=args.window_r,
            device=device,
            dose_per_tilt=args.dose_per_tilt,
            angle_per_tilt=args.angle_per_tilt,
        )
    Nimg = data.N
    if ti_cfg.get('multi', None) is not None:
        Nimg *= ti_cfg['multi']
    D = data.D
    
    # load ctf
    if args.ctf is not None:
        if args.use_real:
            raise NotImplementedError(
                "Not implemented with real-space encoder. Use phase-flipped images instead"
            )
        logger.info("Loading ctf params from {}".format(args.ctf))
        ctf_params = ctf.load_ctf_for_training(D - 1, args.ctf)
        args.Apix = ctf_params[0][0]
        if args.ind is not None:
            ctf_params = ctf_params[ind, ...]
        assert ctf_params.shape == (Nimg, 8)
        if args.encode_mode == "tilt":  # TODO: Parse this in cryodrgn parse_ctf_star
            ctf_params = np.concatenate(
                (ctf_params, data.ctfscalefactor.reshape(-1, 1)), axis=1  # type: ignore
            )
            data.voltage = float(ctf_params[0, 4])
        ctf_params = torch.tensor(ctf_params)  # Nx8
    else:
        ctf_params = None
        args.Apix = 4.5 # HACK, fix later
    all_ctf_params = ctf_params, None
    
    # create lattice
    lattice = Lattice(D, extent=0.5)
    if args.enc_mask is None:
        args.enc_mask = D // 2
    if args.enc_mask > 0:
        assert args.enc_mask <= D // 2
        enc_mask = lattice.get_circular_mask(args.enc_mask)
        in_dim = int(enc_mask.sum())
    elif args.enc_mask == -1:
        enc_mask = None
        in_dim = lattice.D**2 if not args.use_real else (lattice.D - 1) ** 2
    else:
        raise RuntimeError(
            "Invalid argument for encoder mask radius {}".format(args.enc_mask)
        )

    print('Is there a gpu:', torch.cuda.is_available())
    ckpt = torch.load(args.load)
    # load poses
    if args.load_poses:
        logger.info("Loading poses from checkpoint {}".format(args.load))
        train_posetracker = PoseTracker(
            (ckpt['state_dict']['train_posetracker.rots'], ckpt['state_dict']['train_posetracker.trans']), Nimg, D, "s2s2", ind,
        )
    else:
        train_posetracker = PoseTracker.load(
            args.poses, Nimg, D, None, ind, 
        )
    # train_posetracker.rots = ckpt['state_dict']['train_posetracker.rots']
    # train_posetracker.trans = ckpt['state_dict']['train_posetracker.trans']
    # try:
    #     all_ctf_params = ckpt['state_dict']['train_ctf_params'], ckpt['state_dict']['val_ctf_params']
    #     args.Apix = all_ctf_params[0][0][0]
    # except:
    #     all_ctf_params = None, None
    #     args.Apix = 3.
    lattice.coords = ckpt['state_dict']['lattice.coords']
    lattice.freqs2d = ckpt['state_dict']['lattice.freqs2d']
    
    # create dataloader
    data_generator = dataset.make_dataloader(
        data,
        batch_size=1,
        num_workers=ti_cfg['train_dataset']['loader']['num_workers'],
        shuffler_size=args.shuffler_size,
        shuffle=False,
    )
    
    # Make Lightning model module and load from checkpoint
    model = Hypernet.load_from_checkpoint(
        args.load,
        args=args, 
        cfg=ti_cfg, 
        logger=False, 
        posetrackers=(None, None), 
        ctf_params=(None, None), 
        lattice=lattice,
        strict=False,
    )
    model.train_posetracker = train_posetracker.to(model.device)
    if ctf_params is not None:
        model.train_ctf_params = ctf_params.to(model.device)
    
    # create volumes and save them
    model.eval()
    new_i = 0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(data_generator), total=len(data_generator)):
            if args.img_list:
                if (i + 1) in list(itertools.accumulate(args.img_list)):
                    batch = [batch[0].to(model.device), batch[1], batch[2]]
                    vol = model.eval_volume(batch)
                    out_mrc = "{}/{}{:03d}.mrc".format(args.outdir, args.prefix, new_i)
                    if args.flip:
                        vol = vol.flip([0])
                    if args.invert:
                        vol *= -1
                    MRCFile.write(
                        out_mrc, np.array(vol.cpu()).astype(np.float32), Apix=args.Apix
                    )
                    new_i += 1
            else:
                if i % args.skip == 0:
                    batch = [batch[0].to(model.device), batch[1], batch[2]]
                    vol = model.eval_volume(batch)
                    out_mrc = "{}/{}{:03d}.mrc".format(args.outdir, args.prefix, new_i)
                    if args.flip:
                        vol = vol.flip([0])
                    if args.invert:
                        vol *= -1
                    MRCFile.write(
                        out_mrc, np.array(vol.cpu()).astype(np.float32), Apix=args.Apix
                    )
                    new_i += 1