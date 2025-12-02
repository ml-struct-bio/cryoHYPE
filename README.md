# cryoHYPE

## Installation 

Install `cryoHYPE` as follows: 

    # Create and activate conda environment
    (base) $ conda create --name cryohype python=3.9
    (cryohype) $ conda activate cryohype

    # install cryohype
    (cryohype) $ cd cryoHYPE 
    (cryohype) $ pip install .

## Training `cryoHYPE`

### Data preparation

In this example, we will run train and evaluate `cryoHYPE` on the Tomotwin-100 dataset. To download the dataset, follow the instructions on the [CryoBench website](https://cryobench.cs.princeton.edu/). Let `PATH_TO_TOMOTWIN` be the path to the Tomotwin-100 dataset.  

### Configuring `cryoHYPE`

A sample config file for training `cryoHYPE` on Tomotwin-100 can be found in the `cfgs` folder.

### Training `cryoHYPE`

`cryoHYPE` is usually trained with 2 GPUs. Training can be done with the following command:

```
python train_hypernetwork_parallel \
    "${PATH_TO_TOMOTWIN}/Tomotwin-100/images/snr0.01/sorted_particles.128.txt" \
    "${PATH_TO_TOMOTWIN}/Tomotwin-100/images/snr0.01/sorted_particles.128.txt" \
    --poses "${PATH_TO_TOMOTWIN}/Tomotwin-100/combined_poses.pkl" \
    --val-poses "${PATH_TO_TOMOTWIN}/Tomotwin-100/combined_poses.pkl" \
    --ctf "${PATH_TO_TOMOTWIN}/Tomotwin-100/combined_ctfs.pkl" \
    --val-ctf "${PATH_TO_TOMOTWIN}/Tomotwin-100/combined_ctfs.pkl" \
    -o [directory where you want to save the results] \
    --cfg cfgs/tt_base.yaml \
```

If you want to track the training of your model, you can use the `-w` flag to enable WandB and provide your WandB setup using the `--wandb-yaml` flag.

### Volume generation

Rendering volumes from a trained `cryoHYPE` checkpoint can be done with the following command:

```
python eval_vol_hypernet_parallel \
    "${PATH_TO_TOMOTWIN}/Tomotwin-100/images/snr0.01/sorted_particles.128.txt" \
    --poses "${PATH_TO_TOMOTWIN}/Tomotwin-100/combined_poses.pkl" \
    --ctf "${PATH_TO_TOMOTWIN}/Tomotwin-100/combined_ctfs.pkl" \
    -o [directory where you want to save the volumes] \
    --cfg cfgs/tt_base.yaml \
    --load [trained cryoHYPE checkpoint] \
    -c [saved config, found in the directory where you saved training results as config.yaml]
```

### Evaluating rendered volumes

FSC curves are computed by the following command:

```
python analysis_scripts/ours_only_fsc.py \
    [directory where you saved the rendered volumes] \
    -o [directory where you want to save FSC results] \
    --gt-dir "${PATH_TO_TOMOTWIN}/Tomotwin-100/vols/128_org" \
    --mask "${PATH_TO_TOMOTWIN}/Tomotwin-100/init_mask/mask.mrc" 
```

FSC curves can be plotted and FSC-AUC metrics computed using the following command:

```
python analysis_scripts/per_image_fsc_plot.py \
    [directory where you saved FSC results] \
    --Apix 4.5 > "[directory where you want to save raw FSCs]/End-to-End-FSC_AUC_mask.txt"
```
