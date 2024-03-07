import torch 
import numpy as np
from GIGICA import joint_loss, GIGICA
from torch.nn.utils import weight_norm as wn
from torch.linalg import norm
import nibabel as nib


a = 0.8
b = 1 - a
EGv = 0.3745672075
ErChuPai = 2 / 3.141592653589793  



def gigicar(FmriMatr, ICRefMax):
    # Convert numpy arrays to PyTorch tensors
    FmriMatr = torch.tensor(FmriMatr, dtype=torch.float64)
    ICRefMax = torch.tensor(ICRefMax, dtype=torch.float64)

    # Extract dimensions
    n, m = FmriMatr.shape
    n2, m2 = ICRefMax.shape

    # Subtract mean from observed data
    #FmriMat = FmriMatr - FmriMatr.mean(dim=1, keepdim=True)
    FmriMat=FmriMatr - torch.tile(torch.mean(FmriMatr,1),(m,1)).T
    # Calculate covariance matrix
    CovFmri = (FmriMat @ FmriMat.t()) / m

    # Perform PCA reduction on signal
    D, E = torch.linalg.eig(CovFmri)

    #D = D[:, 0].real if len(D.shape) > 1 else D.real  # Extract real parts of eigenvalues
    EsICnum = ICRefMax.shape[0]
    D = D.real
    # Sort eigenvalues and eigenvectors
    index = D.argsort()
    eigenvalues = D[index]
    cols=E.shape[1]
    Esort=torch.zeros(E.shape)
    dsort=torch.zeros(eigenvalues.shape)
    for i in range(cols):
        Esort[:,i] = E[:,index[cols-i-1] ]
        dsort[i]   = D[index[cols-i-1] ]


    thr = 0  # Set your threshold value here
    numpc = (dsort > thr).sum()

    # Perform PCA for selected components
    Epart = Esort[:, :numpc]#.real
    dpart = dsort[:numpc]
    Lambda_part = torch.diag(dpart)#.real
    # Whitening source signal
    tmp = torch.sqrt(Lambda_part)
    Lambda_inv = torch.linalg.inv(torch.sqrt(Lambda_part)) 
    WhitenMatrix = Lambda_inv @ Epart.t()
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
    iternum = 100
    a = 0.8
    b = 1 - a
    ICOutMax = torch.zeros((EsICnum, m))
    gradients = []
    for ICnum in range(1):
        print("component: "+str(ICnum))
        reference = ICRefMaxN[ICnum, :]
        wc = (reference @ torch.linalg.pinv(Y)).t()
        wc = wc / norm(wc)
        last_sources = wc.t() @ Y
        EyrInitial = (1 / m) * (last_sources) @ reference.t()
        NegeInitial = nege(last_sources)
        mag_norm = (torch.tan((EyrInitial * 3.141592653589793) / 2)) / NegeInitial
        itertime = 1
        Nemda = .001
        GICA = GIGICA(wc,mag_norm,m)
        
        
        optimizer = torch.optim.SGD(GICA.parameters(), lr=Nemda)
        for i in range(iternum):
            GICA.W.train(True)
            optimizer.zero_grad()
            sources = GICA(Y.t())
            loss = joint_loss(sources, mag_norm, reference.reshape([-1,1]), m, a, b).sum()
            loss.backward()
            print("loss ",i)
            print(loss)
            optimizer.step()
            itertime = itertime + 1
        wx = GICA.W.state_dict()['weight']
        Source = wx @ Y
        gradients.append(np.array(grrs))
        ICOutMax[ICnum, :] = Source.squeeze()
    np.save(f'{output_dir}/{sub_id}_grads.npy',np.array(gradients))
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


########################################################################
# CALL FUNCTIONS
########################################################################

#sub_id = sys.argv[1]
sub_id = "001312269620"
#sub_id = 'test'
print(f"sub_id:{sub_id}")

#func_file = sys.argv[2]
func_file = '/data/qneuromark/Data/FBIRN/ZN_Neuromark/ZN_Prep_fMRI/'+sub_id+'/SM.nii'
print(f"func_file:{func_file}")

#output_dir = sys.argv[3]
output_dir = './out/'
print(f"output_dir:{output_dir}")

#mask_file = sys.argv[4]
mask_file = '/data/users2/jwardell1/nshor_docker/examples/fbirn-project/FBIRN/group_mean_masks/mask_resampled.nii'
print(f"mask_file:{mask_file}")

#template_file = sys.argv[5]
template_file = '/data/users2/jwardell1/ica-torch-gica/sa_script_work/gica/group_level_analysis/Neuromark_fMRI_1.0.nii'
print(f"template_file:{template_file}")




# Load images
src_img = nib.load(func_file)
src_data = torch.tensor(src_img.get_fdata(), dtype=torch.float64)

ref_img = nib.load(template_file)
ref_data = torch.tensor(ref_img.get_fdata(), dtype=torch.float64)

mask_img = nib.load(mask_file)
mask_data = torch.tensor(mask_img.get_fdata(), dtype=torch.float64)

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


TCMax = TCMax.cpu().numpy()
# Save time courses file
tcfilename = f'{output_dir}/{sub_id}_TCMax_Torchified.npy'
np.save(tcfilename, TCMax)

# Reconstruct brain voxels
xdim, ydim, zdim = mask_data.shape
n_comp = ICOutMax.shape[0]
image_stack = torch.zeros((xdim, ydim, zdim, n_comp))

# Convert idx to numpy for compatibility with existing code
idx_np = idx.cpu().numpy()

image_stack[idx_np[0], idx_np[1], idx_np[2], :] = ICOutMax.t()

# Save as nifti
print("saving")
nifti_img = nib.Nifti1Image(image_stack.numpy(), affine=mask_img.get_qform())
nifti_img.header.set_sform(mask_img.header.get_sform(), code=mask_img.get_qform('code')[1])
nifti_img.header.set_qform(mask_img.header.get_qform(), code=mask_img.get_qform('code')[1])
nifti_file = f'{output_dir}/{sub_id}_ICOutMax_Torchified_SQ.nii.gz'
nib.save(nifti_img, nifti_file)