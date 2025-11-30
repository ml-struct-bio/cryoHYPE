# cryoHYPE

## Installation 

Install `cryoHYPE` as follows: 

    # Create and activate conda environment
    (base) $ conda create --name cryohype python=3.10
    (cryohype) $ conda activate cryohype

    # install cryohype
    (cryohype) $ cd cryoHYPE 
    (cryohype) $ pip install -e . 
    (cryohype) $ python -m pip install lightning
    (cryohype) $ pip install einops torchvision wandb

## Training `cryoHYPE`

### Data preparation

In this example, we will run train and evaluate `cryoHYPE` on the Tomotwin-100 dataset. To download the dataset, follow the instructions on the [CryoBench website](https://cryobench.cs.princeton.edu/). Let `TOMOTWIN_DIR` be the path to the Tomotwin-100 dataset.  

