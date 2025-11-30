import torch


def batched_linear_mm(x, wb):
    # x: (B, N, D1); wb: (B, D1 + 1, D2) or (D1 + 1, D2)
    one = torch.ones(*x.shape[:-1], 1, device=x.device)
    if len(x.shape) == len(wb.shape):
        return torch.matmul(torch.cat([x, one], dim=-1), wb)
    elif len(x.shape) < len(wb.shape):
        return torch.matmul(torch.cat([x, one], dim=-1).unsqueeze(1), wb) # no repeat nec. bc broadcasting
    elif len(x.shape) > len(wb.shape):
        return torch.matmul(torch.cat([x, one], dim=-1), wb.unsqueeze(1)) # no repeat nec. bc broadcasting
