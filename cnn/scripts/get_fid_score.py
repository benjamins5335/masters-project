import torch
from pytorch_fid import fid_score

def get_fid_score():
    """
    Calculates FID score between 2 datasets
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fid = fid_score.calculate_fid_given_paths(['data/test', 'data/train'], batch_size=50, device=device, dims=2048, num_workers=4)


if __name__ == "__main__":
    fid = get_fid_score()
    print('FID: {}'.format(fid))
