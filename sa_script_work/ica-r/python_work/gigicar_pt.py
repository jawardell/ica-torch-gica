import torch
import torch.nn.functional as F
from torch.linalg import norm
import sys
import os
import nibabel as nib
import logging
import datetime
from scipy.linalg import sqrtm


def gigicar(FmriMatr, ICRefMax):
    # Convert numpy arrays to PyTorch tensors
    FmriMatr = torch.tensor(FmriMatr, dtype=torch.float32)
    ICRefMax = torch.tensor(ICRefMax, dtype=torch.float32)

    # Extract dimensions
    n, m = FmriMatr.shape
    n2, m2 = ICRefMax.shape

    # Subtract mean from observed data
    FmriMat = FmriMatr - FmriMatr.mean(dim=1, keepdim=True)
    print(f'FmriMat.shape {FmriMat.shape}')

    # Calculate covariance matrix
    CovFmri = FmriMat @ FmriMat.t() / m
    print(f'CovFmri.shape {CovFmri.shape}')
    print(CovFmri.shape)

    # Perform PCA reduction on signal
    D, E = torch.linalg.eig(CovFmri)
    print(f'D.shape {D.shape}')
    print(f'E.shape {E.shape}')

    #D = D[:, 0].real if len(D.shape) > 1 else D.real  # Extract real parts of eigenvalues
    EsICnum = ICRefMax.shape[0]
    D = D.real
    # Sort eigenvalues and eigenvectors
    eigenvalues, index = D.sort(descending=True)
    Esort = E[:, index]
    dsort = eigenvalues


    thr = 0  # Set your threshold value here
    numpc = (dsort > thr).sum()

    # Perform PCA for selected components
    Epart = Esort[:, :numpc].real
    dpart = dsort[:numpc]
    Lambda_part = torch.diag(dpart).real
    print(Epart.t().shape)
    print(type(Epart.t()))
    # Whitening source signal
    WhitenMatrix =  torch.linalg.inv(torch.sqrt(Lambda_part)) @ Epart.t()
    Y = WhitenMatrix @ FmriMat
    if thr<1e-10 and numpc<n:
        for i in range(Y.shape[0]):
            Y[i,:]=Y[i,:]/torch.std(Y[i,:])
    # Normalize source signal
    #Y = F.normalize(Y, dim=1)
    # Normalize reference signals
    ICRefMaxC=ICRefMax - torch.tile(torch.mean(ICRefMax,1), (m2, 1)).T
    ICRefMaxN=torch.zeros((EsICnum,m2))
    for i in range(EsICnum):
        ICRefMaxN[i,:]=ICRefMaxC[i,:]/torch.std(ICRefMaxC[i,:])
    #ICRefMaxN = (ICRefMax - ICRefMax.mean(dim=1, keepdim=True)) / ICRefMax.std(dim=1, keepdim=True)

    # Computing negentropy
    NegeEva = torch.zeros((EsICnum, 1))
    for i in range(EsICnum):
        NegeEva[i] = nege(ICRefMaxN[i, :])

    iternum = 10
    a = 0.5
    b = 1 - a
    EGv = 0.3745672075
    ErChuPai = 2 / 3.141592653589793

    ICOutMax = torch.zeros((EsICnum, m))
    logger.info("Starting with EGv=%f" % EGv)
    for ICnum in range(EsICnum):
        logger.info('gigicar component: %d/%d' % (ICnum, EsICnum))
        reference = ICRefMaxN[ICnum, :]
        wc = (reference @ torch.linalg.pinv(Y)).t()
        wc = wc / norm(wc)
        y1 = wc.t() @ Y
        EyrInitial = (1 / m) * (y1) @ reference.t()
        NegeInitial = nege(y1)
        c = (torch.tan((EyrInitial * 3.141592653589793) / 2)) / NegeInitial
        IniObjValue = a * ErChuPai * torch.arctan(c * NegeInitial) + b * EyrInitial

        itertime = 1
        Nemda = 1
        for i in range(iternum):
            Cosy1 = torch.cosh(y1)
            logCosy1 = torch.log(Cosy1)
            EGy1 = logCosy1.mean()
            Negama = EGy1 - EGv
            dim = y1.shape[0] if len(y1.shape) > 1 else y1.shape[0]
            EYgy = (1 / m) * Y @ (torch.tanh(y1)).t()
            Jy1 = (EGy1 - EGv)**2
            KwDaoshu = ErChuPai * c * (1 / (1 + (c * Jy1)**2))
            Simgrad = (1 / m) * Y @ reference.t()
            g = a * KwDaoshu * 2 * Negama * EYgy + b * Simgrad#.view(Simgrad.shape[0], 1)
            g_norm = torch.sqrt(g.T@g)#torch.linalg.norm(g)
            d = g / g_norm
            wx = wc + Nemda * d #wc.view(wc.shape[0], 1)
            wx = wx / norm(wx)
            y3 = wx.t() @ Y
            PreObjValue = a * ErChuPai * torch.arctan(c * nege(y3)) + b * (1 / m) * y3 @ reference.t()
            ObjValueChange = PreObjValue - IniObjValue
            ftol = 0.02
            dg = g.t() @ d
            print(dg.shape)
            print(g.shape)
            print(d.shape)
            ArmiCondiThr = Nemda * ftol * dg
            print(f"ObjValueChange: {ObjValueChange.shape}")
            print(f'ArmiCondiThr: {ArmiCondiThr.shape}')
            if ObjValueChange < ArmiCondiThr:
                Nemda = Nemda / 2
                continue
            if torch.allclose(wc.t() @ wx, torch.zeros(1), atol=1e-5):
                break
            elif itertime == iternum:
                break
            IniObjValue = PreObjValue
            y1 = y3
            wc = wx
            itertime = itertime + 1

        Source = wx.t() @ Y
        ICOutMax[ICnum, :] = Source

    TCMax = (1 / m) * FmriMatr @ ICOutMax.t()
    return ICOutMax, TCMax


def nege(x):
    y = torch.log(torch.cosh(x))
    E1 = y.mean()
    E2 = 0.3745672075
    return (E1 - E2)**2


########################################################################
# SETUP LOGGER
########################################################################



DEFAULT_REFERENCE_FN = 'pooled_47.nii'
DEFAULT_EXAMPLE_FN = 'example.nii'
FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(filename='pygigicar.log',level=logging.INFO)
logging.basicConfig(format=FORMAT)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)


logger = logging.getLogger('pygigicar')
logger.setLevel(logging.INFO)
# add ch to logger
logger.handlers = []
logger.addHandler(ch)


########################################################################
# CALL FUNCTIONS
########################################################################
'''
if len(sys.argv) != 6:
    print("Usage: python gigicar.py sub_id func_file out_dir mask_file template_file ")
    print(sys.argv)
    sys.exit()
'''
#sub_id = sys.argv[1]
sub_id = '000300655084'
#sub_id = 'test'
print(f"sub_id:{sub_id}")

#func_file = sys.argv[2]
func_file = '/data/users2/jwardell1/nshor_docker/examples/fbirn-project/FBIRN/000300655084/ses_01/processed/func_resampled.nii'
print(f"func_file:{func_file}")

#output_dir = sys.argv[3]
output_dir = '/data/users2/jwardell1/nshor_docker/examples/fbirn-project/FBIRN/000300655084/ses_01/processed'
print(f"output_dir:{output_dir}")

#mask_file = sys.argv[4]
mask_file = '/data/users2/jwardell1/nshor_docker/examples/fbirn-project/FBIRN/group_mean_masks/mask_resampled.nii'
print(f"mask_file:{mask_file}")

#template_file = sys.argv[5]
template_file = '/data/users2/jwardell1/ica-torch-gica/sa_script_work/gica/group_level_analysis/Neuromark_fMRI_1.0.nii'
print(f"template_file:{template_file}")

'''
if not os.path.isfile(func_file):
    print("Error: subject's preprocessed fMRI file not found.")
    sys.exit()


if not os.path.isdir(output_dir):
    print("Error: output dir not found.")
    sys.exit()


if not os.path.isfile(mask_file):
    print("Error: mask file not found.")
    sys.exit()


if not os.path.isfile(template_file):
    print("Error: template file not found.")
    sys.exit()

'''




# Load images
src_img = nib.load(func_file)
src_data = torch.tensor(src_img.get_fdata(), dtype=torch.float32)

ref_img = nib.load(template_file)
ref_data = torch.tensor(ref_img.get_fdata(), dtype=torch.float32)

mask_img = nib.load(mask_file)
mask_data = torch.tensor(mask_img.get_fdata(), dtype=torch.float32)

# Create idx tensor
idx = torch.nonzero(mask_data).t()

# Mask source and reference images
src_data = src_data[idx[0], idx[1], idx[2], :].t()
print(f'src_data.shape {src_data.shape}')

ref_data = ref_data[idx[0], idx[1], idx[2], :].t()
print(f'ref_data.shape {ref_data.shape}')

# Convert idx to numpy for compatibility with existing code
idx_np = idx.cpu().numpy()

# Continue with the rest of your PyTorch code...
ICOutMax, TCMax = gigicar(src_data, ref_data)


import numpy as np
TCMax = TCMax.cpu().numpy()
# Save time courses file
tcfilename = f'{output_dir}/{sub_id}_TCMax_PyTorch.npy'
np.save(tcfilename, TCMax)

# Reconstruct brain voxels
xdim, ydim, zdim = mask_data.shape
n_comp = ICOutMax.shape[0]
image_stack = torch.zeros((xdim, ydim, zdim, n_comp))

# Convert idx to numpy for compatibility with existing code
idx_np = idx.cpu().numpy()

image_stack[idx_np[0], idx_np[1], idx_np[2], :] = ICOutMax.t()

# Save as nifti
nifti_img = nib.Nifti1Image(image_stack.numpy(), affine=mask_img.get_qform())
nifti_img.header.set_sform(mask_img.header.get_sform(), code=mask_img.get_qform('code')[1])
nifti_img.header.set_qform(mask_img.header.get_qform(), code=mask_img.get_qform('code')[1])
nifti_file = f'{output_dir}/{sub_id}_ICOutMax_PyTorch.nii.gz'
nib.save(nifti_img, nifti_file)
