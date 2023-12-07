import torch
from pytorch_fid import fid_score

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fid = fid_score.calculate_fid_given_paths(['data_copy_temp/real', 'data_copy_temp/fake'], batch_size=50, device=device, dims=2048, num_workers=4)
    print('FID: {}'.format(fid))