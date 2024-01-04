import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


sm_img = nib.load('output_data/group_SM.nii.gz')
sm_data = sm_img.get_fdata()
xdim,ydim,zdim,ncomp = sm_data.shape
nvox = xdim*ydim*zdim

mask_img = nib.load('mask.nii')
idx = np.where(mask_img.dataobj)

brain_vox = sm_data[*idx, :]

plots = []
row = 10
col = 10

voxels  = np.array(range(brain_vox.shape[0]))

ix = 0
#fig, ax = plt.subplots(nrows=row, ncols=col, sharex=False, sharey=False,)
for i in range(row):
  for j in range(col):
    if(ix >= ncomp): 
      break
    sm = brain_vox[:,ix ]
    
    
    #ax[i,j].plot(sm)
    filename="comp{}.png".format(ix)
    plt.plot(sm)
    plt.savefig(filename)
    plt.clf()
    ix += 1

#fig.savefig('plots.png')
