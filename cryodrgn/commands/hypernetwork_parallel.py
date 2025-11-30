import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from torchmetrics.functional.image import structural_similarity_index_measure, learned_perceptual_image_patch_similarity

import einops
import numpy as np

from cryodrgn import ctf
from cryodrgn import fft

import cryodrgn.commands.models as models
import cryodrgn.commands.utils as utils

from cryodrgn.frc import frc

class Hypernet(L.LightningModule):
    def __init__(self, args, cfg, logger, posetrackers=None, ctf_params=None, lattice=None):
        super().__init__()
        self.args = args
        self.cfg = cfg
        self.wandb_logger = logger

        self.train_posetracker = posetrackers[0]
        self.val_posetracker = posetrackers[1]

        self.register_buffer('train_ctf_params', ctf_params[0])
        self.register_buffer('val_ctf_params', ctf_params[1])
        self.lattice = lattice

        # Make model
        self.model = models.make(self.cfg['model'], load_sd=False)

    def forward(self, batch):
        '''
        Takes a batch of particle views and returns the parameters of INRs for those particles.
        '''
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        '''
        Full training step of hypernetwork INR training.
        '''
        return self._iter_step(batch, is_train=True)

    def validation_step(self, batch, batch_idx):
        '''
        Full validation step of the hypernetwork which produces metrics at the end.
        '''
        return self._iter_step(batch, is_train=False)

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = utils.make_optimizer(self.model.parameters(), self.cfg['optimizer'])
        scheduler = utils.make_scheduler(optimizer, self.cfg['scheduler'])
        optimizer_dict = {'optimizer': optimizer}
        if self.args.do_pose_sgd:
            optimizer_dict['pose_optimizer'] = torch.optim.SparseAdam(list(self.train_posetracker.parameters()), lr=self.args.pose_lr)
        if scheduler is not None: 
            optimizer_dict['lr_scheduler'] = scheduler
        return optimizer_dict

    def _preprocess_input(self, y, lattice, trans):
        B = y.size(0)
        D = lattice.D
        if len(trans.shape) == 2:
            y = lattice.translate_ht(y.view(B, -1), trans.unsqueeze(1)).view(B, D, D)
        elif len(trans.shape) == 3:
            V = y.shape[1]
            y = einops.rearrange(y, 'b v h w -> (b v) h w')
            trans = einops.rearrange(trans, 'b v d -> (b v) d')
            y = lattice.translate_ht(y.view(B*V, -1), trans.unsqueeze(1)).view(B, V, D, D)
        return y

    def _iter_step(self, batch, is_train):
        # get indices and input particles (y)
        ind = batch[-1] # shape [B] or shape [B, views]
        y = batch[0] # shape [B, D, D], usually [B, 129, 129], or [B, views, D, D]
#         if self.cfg.get('shot', 1) > 1:
#             y = einops.rearrange(y, '(b v) h w -> b v h w', v=self.cfg['shot'])
#             ind = einops.rearrange(ind, '(b v) -> b v', v=self.cfg['shot'])

        # real input
        if self.cfg.get('real', False) or self.cfg.get('both', False):
            real_y = fft.batch_ihtn_center_loop(y)
        
        # set posetracker and ctf_params based on train/val
        if hasattr(self.args, 'do_pose_sgd') and self.args.do_pose_sgd:
            posetracker = self.train_posetracker
        else:
            posetracker = self.train_posetracker if is_train else self.val_posetracker
        ctf_params = self.train_ctf_params if is_train else self.val_ctf_params

        # if operating in real domain, create yr
        yr = None
        if self.args.use_real:
            assert hasattr(data, "particles_real")
            yr = torch.from_numpy(data.particles_real[ind.numpy()])  # type: ignore  # PYR02

        # if tilt series, get rot, tran, ctf_param, and dose_filters; otherwise, get rot, tran, ctf_param
        dose_filters = None
        if self.args.encode_mode == "tilt":
            tilt_ind = minibatch[1]
            assert all(tilt_ind >= 0), tilt_ind
            rot, tran = self.posetracker.get_pose(tilt_ind.view(-1))
            ctf_param = (
                self.ctf_params[tilt_ind.view(-1)] if ctf_params is not None else None
            )
            y = y.view(-1, D, D)
            Apix = self.ctf_params[0, 0] if ctf_params is not None else None
            if self.args.dose_per_tilt is not None:
                dose_filters = data.get_dose_filters(tilt_ind, lattice, Apix)
        else:
            rot, tran = posetracker.get_pose(ind) # (B, 3, 3) or (B, V, 3, 3) | (B, 2) or (B, views, 2)
            ctf_param = ctf_params[ind] if ctf_params is not None else None # shape (B, 8)/(B, views, 8)

        # preprocess y
        if tran is not None:
            y = self._preprocess_input(y, self.lattice, tran)
            
            # real input
            if self.cfg.get('real', False) or self.cfg.get('both', False):
                real_y = self._preprocess_input(real_y, self.lattice, tran)

        use_ctf = ctf_param is not None
        B = y.size(0)
        D = self.lattice.D
        c = None

        # compute ctf
        if use_ctf:
            if len(ctf_param.shape) == 2:
                freqs = self.lattice.freqs2d.unsqueeze(0).expand(
                    B, *self.lattice.freqs2d.shape
                ) / ctf_param[:, 0].view(B, 1, 1)
                c = ctf.compute_ctf(freqs, *torch.split(ctf_param[:, 1:], 1, 1)).view(B, D, D)
            elif len(ctf_param.shape) == 3:
                V = y.shape[1]
                ctf_tmp = ctf_param.view(B*V, -1)
                freqs = self.lattice.freqs2d.unsqueeze(0).expand(
                    B*V, *self.lattice.freqs2d.shape
                ) / ctf_tmp[:, 0].view(B*V, 1, 1) 
                c = ctf.compute_ctf(freqs, *torch.split(ctf_tmp[:, 1:], 1, 1)).view(B, V, D, D)

        # final data preparation, including phase flip by CTF if we have it
        if yr is not None:
            input_ = (yr,)
        else:
            input_ = (y,)
            if c is not None:
                if self.cfg.get('phase_flip', True):
                    input_ = (x * c.sign() for x in input_)  # phase flip by the ctf
                    if self.cfg.get('real', False) or self.cfg.get('both', False):
                        real_y = fft.batch_htn_center_loop(real_y)
                        real_y = real_y * c.sign()
                        real_y = fft.batch_ihtn_center_loop(real_y)
                else:
                    input_ = (x * c for x in input_) # multiply by ctf
                    if self.cfg.get('real', False) or self.cfg.get('both', False):
                        real_y = fft.batch_htn_center_loop(real_y)
                        real_y = real_y * c.sign()
                        real_y = fft.batch_ihtn_center_loop(real_y)

        # above is same as cryodrgn
        
        # mask inputs to 0 if not in mask
        if self.cfg.get('mask_inputs', False) is True:
            mask = self.lattice.get_circular_mask(D // 2) # shape (16641,)
            mask = mask.view(1, D, D)
            input_ = (torch.where(mask.to(x.device), x, 0.) for x in input_)
        
        # reshape input from [B * N, D, D] -> [B, N, 1, D, D] (channels=1)
        if self.cfg['model']['name'] in ['trans_inr_mlp']:
            mask = self.lattice.get_circular_mask(D // 2)
            input_ = (x.view(B, -1)[:, mask] for x in input_)
        else:
            if len(y.shape) == 3:
                input_ = (einops.rearrange(
                    x, '(b n) h w -> b n h w', n=self.cfg.get('shot', 1)
                ).unsqueeze(2) for x in input_)
            
                # real input
                if self.cfg.get('real', False) or self.cfg.get('both', False):
                    real_y = einops.rearrange(
                        real_y, '(b n) h w -> b n h w', n=self.cfg.get('shot', 1)
                    ).unsqueeze(2)
            elif len(y.shape) == 4:
                input_ = (x.unsqueeze(2) for x in input_)
            
                # real input
                if self.cfg.get('real', False) or self.cfg.get('both', False):
                    real_y = real_y.unsqueeze(2)

        # if crop, crop the center of the FT of the image
        if self.cfg.get('crop', 0) > 0:
            start = D//2 - self.cfg['crop']//2
            end = start + self.cfg['crop'] + 1
            input_ = (x[..., start:end, start:end] for x in input_)
            
        # append pose and ctf to input if we are also tokenizing them (hardcode for now)
        if self.cfg.get('tokenize', False):
            pose = torch.cat([rot.view(B, -1), tran], dim=-1)
            input_ = list(input_) + [pose, c]
            
        # add adaptive noise
        if self.cfg.get('add_noise', 0) > 0:
            input_ = (x + self.cfg['add_noise']/2 * torch.randn(*x.shape).to(x.device) for x in input_)

        # handle real input and real + fourier input
        if self.cfg.get('real', False):
            input_ = list(input_)
            input_[0] = real_y
        elif self.cfg.get('both', False):
            input_ = list(input_)
            input_[0] = torch.cat([input_[0], real_y], dim=2)

        # run hypernetwork to get INR
        # yr = torch.from_numpy(self.data.particles_real[ind.numpy()])  # type: ignore  # PYR02
        # input_real = (yr,)
        # input_real = (real_y,)
        if self.cfg['model']['name'] in ['trans_inr_cat', 'trans_inr_local', 'trans_inr_autoenc', 'trans_inr_rep', 'trans_inr_lcond']:
            hyponet, z = self.model(*input_)
        else:
            hyponet = self.model(*input_)

        # create coordinate mask so we can get coords for INR/hyponet
        if self.cfg.get('mask', True):
            mask_size = D // 2
            if self.cfg.get('frequency_marching', None) is not None:
                mask_size = self.cfg['frequency_marching']
            mask = self.lattice.get_circular_mask(mask_size) # shape (16641,)

        # input coordinates to INR/hyponet
        if self.cfg.get('mask', True):
            coords = self.lattice.coords[mask] / self.lattice.extent / 2 @ rot # something like [B, 12852,3] or [B, views, 12852, 3]
        else:
            coords = self.lattice.coords / self.lattice.extent / 2 @ rot
        
        # subsample to increase batch size 
        if self.cfg.get('subsample', None) is not None:
            coords_idxs = np.random.randint(coords.shape[-2], size=self.cfg['subsample'])
            coords = coords[..., coords_idxs, :] # shape 
        if len(y.shape) == 3:
            if self.cfg['model']['name'] in ['trans_inr_lcond']:
                y_recon = hyponet(coords, z).view(B, -1)
            else:
                y_recon = hyponet(coords).view(B, -1)
        else:
            if self.cfg['model']['name'] in ['trans_inr_cat', 'trans_inr_local', 'trans_inr_autoenc', 'trans_inr_lcond']:
                y_recon = hyponet(coords, z)
            else:
                y_recon = hyponet(coords)

        # apply CTF if we have it
        if c is not None:
            effective_B = B * c.shape[1] if len(c.shape) == 4 else B
            if self.cfg.get('mask', True):
                if self.cfg.get('subsample', None) is not None:
                    y_recon *= c.view(effective_B, -1)[:, mask][:, coords_idxs]
                else:
                    y_recon *= c.view(effective_B, -1)[:, mask]
            else:
                y_recon *= c.view(effective_B, -1)

        # mask y, apply dose filter if it exists, and compute MSE
        if len(y.shape) == 3:
            y = y.view(B, -1)
        elif len(y.shape) == 4:
            y = einops.rearrange(y, 'b v n d -> (b v) (n d)')
        if self.cfg.get('mask', True):
            y = y[:, mask]
        if self.cfg.get('subsample', None) is not None:
            y = y[:, coords_idxs]
        if dose_filters is not None:
            y_recon = torch.mul(y_recon, dose_filters[:, mask])
        loss = F.mse_loss(y_recon, y)
        if self.cfg.get('l2_reg_dec', 0) > 0:
            l2_reg = 0.
            for name, param in hyponet.params.items():
                l2_reg += torch.sum(param**2)
            loss += self.cfg['l2_reg_dec'] * l2_reg
        if self.cfg.get('frc_weight', 0) > 0 and not self.cfg.get('mask', True):
            # reshape from [B, D^2] to [B, D, D]
            y_recon = y_recon.view(-1, D, D) # y_recon should be shape [B, D, D] now
            y = y.view(-1, D, D) # y should be shape [B, D, D] now
            loss += self.cfg['frc_weight'] * frc(y_recon, y)
        if self.cfg.get('smoothness_weight', 0) > 0:
            # y_recon_real = fft.batch_ihtn_center_loop(y_recon.view(B, D, D)[:, :-1, :-1])
            # loss += self.cfg['smoothness_weight'] * y_recon_real.pow_(2).sum()
            loss += self.cfg['smoothness_weight'] * torch.square(y_recon).sum()
            # if self.cfg.get('l2_real', False):
            #     y_real = fft.batch_ihtn_center_loop(y_recon.view(B, D, D)[:, :-1, :-1])
            #     loss += self.cfg['smoothness_weight'] * y_real.pow_(2).mean()
            # else:
            #     loss += self.cfg['smoothness_weight'] * torch.square(y_recon).mean()
        if self.cfg.get('inv_weight', 0) > 0:
            loss += self.cfg['inv_weight'] * F.mse_loss(z[0], z[1])
        if self.cfg.get('ssim', 0) > 0 and not self.cfg.get('mask', True):
            # reshape from [B, D^2] to [B, 3, D, D]
            y_recon = y_recon.view(-1, D, D).unsqueeze(1).repeat(1, 3, 1, 1) # y_recon should be shape [B, 3, D, D] now
            y = y.view(-1, D, D).unsqueeze(1).repeat(1, 3, 1, 1) # y should be shape [B, 3, D, D] now
            
            # compute SSIM loss
            loss += self.cfg['ssim'] * structural_similarity_index_measure(y_recon, y)
        if self.cfg.get('lpips', 0) > 0 and not self.cfg.get('mask', True):
            # normalize y_recon, y to only have values in [0, 1]
            y_recon -= y_recon.min(1, keepdim=True)[0]
            y_recon /= y_recon.max(1, keepdim=True)[0]
            y -= y.min(1, keepdim=True)[0]
            y /= y.max(1, keepdim=True)[0]
            
            # reshape from [B, D^2] to [B, 3, D, D]
            y_recon = y_recon.view(-1, D, D).unsqueeze(1).repeat(1, 3, 1, 1) # y_recon should be shape [B, 3, D, D] now
            y = y.view(-1, D, D).unsqueeze(1).repeat(1, 3, 1, 1) # y should be shape [B, 3, D, D] now
            
            # compute LPIPS loss
            loss += self.cfg['lpips'] * learned_perceptual_image_patch_similarity(y_recon, y, normalize=True)

        # log to wandb
        prefix = 'train' if is_train else 'test'
        self.log(prefix + '/loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        
        return {'loss': loss}
    
    def _forward(self, batch, is_train, get_z=False):
        # get indices and input particles (y)
        ind = batch[-1] # shape [B]
        y = batch[0] # shape [B, D, D], usually [B, 129, 129]
        
        # real input
        if self.cfg.get('real', False) or self.cfg.get('both', False):
            real_y = fft.batch_ihtn_center_loop(y)
        
        # set posetracker and ctf_params based on train/val
        posetracker = self.train_posetracker if is_train else self.val_posetracker
        ctf_params = self.train_ctf_params if is_train else self.val_ctf_params

        # if operating in real domain, create yr
        yr = None
        if self.args.use_real:
            assert hasattr(data, "particles_real")
            yr = torch.from_numpy(data.particles_real[ind.numpy()])  # type: ignore  # PYR02

        # if tilt series, get rot, tran, ctf_param, and dose_filters; otherwise, get rot, tran, ctf_param
        dose_filters = None
        if self.args.encode_mode == "tilt":
            tilt_ind = minibatch[1]
            assert all(tilt_ind >= 0), tilt_ind
            rot, tran = self.posetracker.get_pose(tilt_ind.view(-1))
            ctf_param = (
                self.ctf_params[tilt_ind.view(-1)] if ctf_params is not None else None
            )
            y = y.view(-1, D, D)
            Apix = self.ctf_params[0, 0] if ctf_params is not None else None
            if self.args.dose_per_tilt is not None:
                dose_filters = data.get_dose_filters(tilt_ind, lattice, Apix)
        else:
            rot, tran = posetracker.get_pose(ind)
            ctf_param = ctf_params[ind] if ctf_params is not None else None

        # preprocess y
        if tran is not None:
            y = self._preprocess_input(y, self.lattice, tran)

            # real input
            if self.cfg.get('real', False) or self.cfg.get('both', False):
                real_y = self._preprocess_input(real_y, self.lattice, tran)

        use_ctf = ctf_param is not None
        B = y.size(0)
        D = self.lattice.D
        c = None

        # compute ctf
        if use_ctf:
            if len(ctf_param.shape) == 2:
                freqs = self.lattice.freqs2d.unsqueeze(0).expand(
                    B, *self.lattice.freqs2d.shape
                ) / ctf_param[:, 0].view(B, 1, 1)
                c = ctf.compute_ctf(freqs, *torch.split(ctf_param[:, 1:], 1, 1)).view(B, D, D)
            elif len(ctf_param.shape) == 3:
                V = y.shape[1]
                ctf_tmp = ctf_param.view(B*V, -1)
                freqs = self.lattice.freqs2d.unsqueeze(0).expand(
                    B*V, *self.lattice.freqs2d.shape
                ) / ctf_tmp[:, 0].view(B*V, 1, 1) 
                c = ctf.compute_ctf(freqs, *torch.split(ctf_tmp[:, 1:], 1, 1)).view(B, V, D, D)

        # final data preparation, including phase flip by CTF if we have it
        if yr is not None:
            input_ = (yr,)
        else:
            input_ = (y,)
            if c is not None:
                if self.cfg.get('phase_flip', True):
                    input_ = (x * c.sign() for x in input_)  # phase flip by the ctf
                    if self.cfg.get('real', False) or self.cfg.get('both', False):
                        real_y = fft.batch_htn_center_loop(real_y)
                        real_y = real_y * c.sign()
                        real_y = fft.batch_ihtn_center_loop(real_y)
                else:
                    input_ = (x * c for x in input_) # multiply by ctf
                    if self.cfg.get('real', False) or self.cfg.get('both', False):
                        real_y = fft.batch_htn_center_loop(real_y)
                        real_y = real_y * c
                        real_y = fft.batch_ihtn_center_loop(real_y)

        # above is same as cryodrgn
        
        # mask inputs to 0 if not in mask
        if self.cfg.get('mask_inputs', False) is True:
            mask = self.lattice.get_circular_mask(D // 2) # shape (16641,)
            mask = mask.view(1, D, D)
            input_ = (torch.where(mask.to(x.device), x, 0.) for x in input_)
        
        # reshape input from [B * N, D, D] -> [B, N, 1, D, D] (channels=1)
        if self.cfg['model']['name'] in ['trans_inr_mlp']:
            mask = self.lattice.get_circular_mask(D // 2)
            input_ = (x.view(B, -1)[:, mask] for x in input_)
        else:
            if len(y.shape) == 3:
                input_ = (einops.rearrange(
                    x, '(b n) h w -> b n h w', n=self.cfg.get('shot', 1)
                ).unsqueeze(2) for x in input_)

                # real input
                if self.cfg.get('real', False) or self.cfg.get('both', False):
                    real_y = einops.rearrange(
                        real_y, '(b n) h w -> b n h w', n=self.cfg.get('shot', 1)
                    ).unsqueeze(2)
            elif len(y.shape) == 4:
                input_ = (x.unsqueeze(2) for x in input_)

                # real input
                if self.cfg.get('real', False) or self.cfg.get('both', False):
                    real_y = real_y.unsqueeze(2)
            
        # if crop, crop the center of the FT of the image
        if self.cfg.get('crop', 0) > 0:
            start = D//2 - self.cfg['crop']//2
            end = start + self.cfg['crop'] + 1
            input_ = (x[..., start:end, start:end] for x in input_)
        
        # append pose and ctf to input if we are also tokenizing them (hardcode for now)
        if self.cfg.get('tokenize', False):
            pose = torch.cat([rot.view(B, -1), tran], dim=-1)
            input_ = list(input_) + [pose, c]

        # handle real input and real + fourier input
        if self.cfg.get('real', False):
            input_ = list(input_)
            input_[0] = real_y
        elif self.cfg.get('both', False):
            input_ = list(input_)
            input_[0] = torch.cat([input_[0], real_y], dim=2)
        
        if get_z:
            out = self.model.get_z(*input_)
        else:
            out = self.model(*input_)
        
        return out
    
    def eval_volume(self, batch):
        """
        Evaluate the model on a DxDxD volume
        """
        # downsample?
        norm = self.args.norm
        if self.args.downsample:
            coords = self.lattice.get_downsample_coords(args.downsample + 1)
            extent = self.lattice.extent * (args.downsample / (self.lattice.D - 1))
            D = args.downsample + 1
        else:
            coords = self.lattice.coords
            extent = self.lattice.extent
            D = self.lattice.D
        
        # Note: extent should be 0.5 by default, except when a downsampled
        # volume is generated
        assert extent <= 0.5
        assert not self.training
        
        # get hyponet
        with torch.no_grad():
            if self.cfg['model']['name'] in ['trans_inr_cat', 'trans_inr_local', 'trans_inr_autoenc', 'trans_inr_rep', 'trans_inr_lcond']:
                hyponet, z = self._forward(batch, is_train=True)
            elif self.cfg['model']['name'] in ['trans_inr_mlp']:
                hyponet = self._forward(batch, is_train=True)
            else:
                hyponet = self._forward(batch, is_train=True)
        
        # intialize volume
        vol_f = torch.zeros((D, D, D), dtype=torch.float32)
        
        # check if slice matches training for debugging purposes
        with torch.no_grad():
            pp = self._iter_step(batch, is_train=True)
            print('Loss: ', pp['loss'])
        
        # evaluate the volume by zslice to avoid memory overflows
        for i, dz in enumerate(np.linspace(-extent, extent, D, endpoint=True, dtype=np.float32)):
            x = coords + torch.tensor([0, 0, dz]).cuda() # HACK HACK HACK
            x = x.unsqueeze(0)
            with torch.no_grad():
                if self.cfg['model']['name'] in ['trans_inr_cat', 'trans_inr_local', 'trans_inr_autoenc']:
                    x = x.unsqueeze(1)
                    y = hyponet(x, z)
                elif self.cfg['model']['name'] in ['trans_inr_lcond']:
                    y = hyponet(x, z)
                elif self.cfg['model']['name'] in ['trans_inr_ipc']:
                    x = x.unsqueeze(0)
                    y = hyponet(x)
                elif self.cfg['model']['name'] in ['trans_inr_mlp']:
                    y = hyponet(x)
                else:
                    y = hyponet(x)
                y = y.view(D, D)
            vol_f[i] = y
        
        # apply normalization
        vol_f = vol_f * norm[1] + norm[0]
        
        # Fourier transform
        vol = fft.ihtn_center(
            vol_f[0:-1, 0:-1, 0:-1]
        )  # remove last +k freq for inverse FFT
        return vol
    
    def eval_z(self, batch):
        return self._forward(batch, is_train=True, get_z=True)