"""
Train a VAE for heterogeneous reconstruction with known pose
"""
import argparse
import os
import pickle
import sys
import logging
from datetime import datetime as dt
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
import torch.nn.functional as F
from torchvision.transforms import v2

try:
    import apex.amp as amp  # type: ignore  # PYR01
except ImportError:
    pass

import yaml
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateFinder, LearningRateMonitor

import cryodrgn
from cryodrgn import __version__, ctf
from cryodrgn import dataset
from cryodrgn.lattice import Lattice
from cryodrgn.pose import PoseTracker
import cryodrgn.config

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
        "val",
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
        "--val-poses", type=os.path.abspath, required=True, help="Image poses (.pkl)"
    )
    parser.add_argument(
        "--ctf", metavar="pkl", type=os.path.abspath, help="CTF parameters (.pkl)"
    )
    parser.add_argument(
        "--val-ctf", metavar="pkl", type=os.path.abspath, help="CTF parameters (.pkl)"
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
        "-v", "--verbose", action="store_true", help="Increase verbosity"
    )
    parser.add_argument(
        "--seed", type=int, default=np.random.randint(0, 100000), help="Random seed"
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
    group.add_argument(
        '--no-validation',
        action='store_true',
        help='whether to perform validation',
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
        "-d",
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

    group = parser.add_argument_group("Pose SGD")
    group.add_argument(
        "--do-pose-sgd", action="store_true", help="Refine poses with gradient descent"
    )
    group.add_argument(
        "--pretrain",
        type=int,
        default=1,
        help="Number of epochs with fixed poses before pose SGD (default: %(default)s)",
    )
    group.add_argument(
        "--emb-type",
        choices=("s2s2", "quat"),
        default="quat",
        help="SO(3) embedding type for pose SGD (default: %(default)s)",
    )
    group.add_argument(
        "--pose-lr",
        type=float,
        default=3e-4,
        help="Learning rate for pose optimizer (default: %(default)s)",
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


def save_config(args, dataset, lattice, out_config):
    dataset_args = dict(
        particles=args.particles,
        norm=dataset.norm,
        invert_data=args.invert_data,
        ind=args.ind,
        keepreal=args.use_real,
        window=args.window,
        window_r=args.window_r,
        datadir=args.datadir,
        ctf=args.ctf,
        poses=args.poses,
        do_pose_sgd=args.do_pose_sgd,
    )
    if args.encode_mode == "tilt":
        dataset_args["ntilts"] = args.ntilts

    lattice_args = dict(D=lattice.D, extent=lattice.extent, ignore_DC=lattice.ignore_DC)
    config = dict(
        dataset_args=dataset_args, lattice_args=lattice_args
    )
    config["seed"] = args.seed
    cryodrgn.config.save(config, out_config)


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
        exp_name = os.path.basename(args.outdir).split('.')[0]
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
    cfg = make_cfg(args)
    
    # Setup wandb
    wandb_logger = False
    if args.wandb_upload:
        with open(args.wandb_yaml, 'r') as f:
            wandb_cfg = yaml.load(f, Loader=yaml.FullLoader)
        # os.environ['WANDB_DIR'] = cfg['env']['save_dir']
        # os.environ['WANDB_NAME'] = cfg['env']['exp_name']
        os.environ['WANDB_API_KEY'] = wandb_cfg['api_key']
        os.environ['WANDB_CACHE_DIR'] = cfg['env']['save_dir']
        wandb_logger = WandbLogger(
            name=cfg['env']['exp_name'],
            save_dir=cfg['env']['save_dir'],
            project=wandb_cfg['project'], 
            entity=wandb_cfg['entity'],
            config=cfg,
        )
        
    # create output directory if it doesn't already exist
    if args.outdir is not None and not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

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
                v2.GaussianNoise(sigma=1.6979357699380342/2),
            ])
        contrastive = False
        if transform and cfg.get('inv_weight', 0) > 0:
            contrastive = True
            cfg['shot'] = 2 * cfg.get('shot', 1)
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
            multi=cfg.get('multi', None),
            shot=cfg.get('shot', 1),
            transform=transform,
            # contrastive=contrastive,
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
    if not args.no_validation:
        if args.encode_mode != "tilt":
            args.use_real = args.encode_mode == "conv"  # Must be False
            if contrastive:
                cfg['shot'] = cfg['shot']//2
            val_data = dataset.ImageDataset(
                mrcfile=args.val,
                lazy=args.lazy,
                norm=args.norm,
                invert_data=args.invert_data,
                ind=ind,
                keepreal=args.use_real,
                window=args.window,
                datadir=args.datadir,
                window_r=args.window_r,
                max_threads=args.max_threads,
                multi=cfg.get('multi', None),
                shot=cfg.get('shot', None),
            )
        else:
            assert args.encode_mode == "tilt"
            val_data = dataset.TiltSeriesData(  # FIXME: maybe combine with above?
                args.val,
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
    if not args.no_validation:
        val_Nimg = val_data.N
    if cfg.get('multi', None) is not None:
        Nimg *= cfg['multi']
        if not args.no_validation:
            val_Nimg *= cfg['multi']
    D = data.D

    # load poses
    # if args.do_pose_sgd:
    #     assert (
    #         args.domain == "hartley"
    #     ), "Need to use --domain hartley if doing pose SGD"
    do_pose_sgd = args.do_pose_sgd
    train_posetracker = PoseTracker.load(
        args.poses, Nimg, D, "s2s2" if do_pose_sgd else None, ind, 
    )
    if args.no_validation:
        val_posetracker = None
    else:
        val_posetracker = PoseTracker.load(
            args.val_poses, val_Nimg, val_data.D, "s2s2" if do_pose_sgd else None, ind, 
        )
    posetrackers = train_posetracker, val_posetracker

    # load ctf
    if args.ctf is not None:
        if args.use_real:
            raise NotImplementedError(
                "Not implemented with real-space encoder. Use phase-flipped images instead"
            )
        logger.info("Loading ctf params from {}".format(args.ctf))
        ctf_params = ctf.load_ctf_for_training(D - 1, args.ctf)
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
    if args.val_ctf is not None and not args.no_validation:
        if args.use_real:
            raise NotImplementedError(
                "Not implemented with real-space encoder. Use phase-flipped images instead"
            )
        logger.info("Loading ctf params from {}".format(args.val_ctf))
        val_ctf_params = ctf.load_ctf_for_training(D - 1, args.val_ctf)
        if args.ind is not None:
            val_ctf_params = val_ctf_params[ind, ...]
        assert val_ctf_params.shape == (val_Nimg, 8)
        if args.encode_mode == "tilt":  # TODO: Parse this in cryodrgn parse_ctf_star
            val_ctf_params = np.concatenate(
                (val_ctf_params, data.ctfscalefactor.reshape(-1, 1)), axis=1  # type: ignore
            )
            data.voltage = float(val_ctf_params[0, 4])
        val_ctf_params = torch.tensor(val_ctf_params)  # Nx8
    else:
        val_ctf_params = None
    all_ctf_params = ctf_params, val_ctf_params

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

    # save configuration
    out_config = "{}/config.yaml".format(args.outdir)
    save_config(args, data, lattice, out_config)

    # create dataloader
    batch_size = cfg['train_dataset']['loader']['batch_size']# * cfg.get('shot', 1)
    data_generator = dataset.make_dataloader(
        data,
        batch_size=batch_size,
        num_workers=cfg['train_dataset']['loader']['num_workers'],
        shuffler_size=args.shuffler_size,
    )
    if args.no_validation:
        val_data_generator = None
    else:
        val_data_generator = dataset.make_dataloader(
            val_data,
            batch_size=batch_size,
            num_workers=cfg['test_dataset']['loader']['num_workers'],
            shuffler_size=args.shuffler_size,
            shuffle=False,
        )

    # same as cryodrgn until here
    # Make Lightning model module
    if args.load:
        model = Hypernet.load_from_checkpoint(
            args.load,
            args=args, 
            cfg=cfg, 
            logger=wandb_logger, 
            posetrackers=(None, None), 
            ctf_params=all_ctf_params, 
            lattice=lattice,
            strict=False,
        )
        model.train_posetracker, model.val_posetracker = posetrackers
    else:
        model = Hypernet(args, cfg, wandb_logger, posetrackers, all_ctf_params, lattice)

    # create callbacks
    save_best = ModelCheckpoint(save_top_k=1, dirpath=args.outdir, monitor='test/loss', mode='min')
    # lr_finder = LearningRateFinder()
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    if cfg.get('callback_type', 'save_best_only') == 'save_best_only':
        save_last = ModelCheckpoint(dirpath=args.outdir, monitor='epoch', mode='max')
        callbacks = [save_best, save_last, lr_monitor]
    elif cfg.get('callback_type', 'save_best_only') == 'save_all':
        save_all = ModelCheckpoint(save_top_k=-1, dirpath=args.outdir, every_n_epochs=1)
        callbacks = [save_best, save_all, lr_monitor]
    elif cfg.get('callback_type', 'save_best_only').startswith('save_every_'):
        save_freq = int(cfg['callback_type'].split('_')[-1])
        save_every = ModelCheckpoint(save_top_k=-1, dirpath=args.outdir, every_n_epochs=save_freq)
        callbacks = [save_best, save_every, lr_monitor]

    # instantiate model
    trainer = L.Trainer(
        **cfg['lightning'],
        logger=wandb_logger,
        default_root_dir=args.outdir,
        callbacks=callbacks,
    )
    trainer.fit(
        model=model, 
        train_dataloaders=data_generator, 
        val_dataloaders=val_data_generator,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
