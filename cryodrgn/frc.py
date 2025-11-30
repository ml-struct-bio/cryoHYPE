import torch

def frc(pred, gt):
    '''
    pred: [B, D, D], already fourier transformed
    gt:   [B, D, D], already fourier transformed
    
    return: avg FRC across all integer frequencies < D//2
    '''
    D = gt.shape[-1]
    x = torch.arange(-D // 2, D // 2)
    x1, x0 = torch.meshgrid(x, x, indexing="ij")
    coords = torch.stack((x0, x1), dim=-1)
    r = (coords**2).sum(-1) ** 0.5
    
    prev_mask = torch.zeros((D, D), dtype=torch.bool)
    frc = 0.0
    for i in range(1, D // 2):
        mask = r < i
        shell = torch.where(mask & torch.logical_not(prev_mask))
        v1 = pred[:, shell[0], shell[1]] # [B, num_pts]
        v2 = gt[:, shell[0], shell[1]] # [B, num_pts]
        num = torch.bmm(v1.unsqueeze(-2), v2.unsqueeze(-1)).squeeze()
        den = torch.sqrt(
            torch.sum(torch.square(v1), dim=-1) * torch.sum(torch.square(v2), dim=-1)
        )
        frc += num/den
        prev_mask = mask
    
    return (frc/(D//2 - 1)).mean()