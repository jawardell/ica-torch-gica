import numpy as np
import sys
import os
import nibabel as nib
import logging
import datetime
from scipy.linalg import sqrtm


def gigicar(FmriMatr,ICRefMax):
	#written by Brad Baker, MRN. 2018.
	#Based on MATLAB code by 

	#Input:
	#FmriMatr is the observed data with size of timepoints*voxelvolums (numpy)
	#ICRefMax includes the reference signals (numpy)

	#Output
	#ICOutMax includes the estimated ICs
	#TCMax is the obtained mixing matrix

	n,m = FmriMatr.shape
	n2,m2 = ICRefMax.shape
	logger.info("\n\nStarting GIGICA at %s" % str(datetime.datetime.now()))
	logger.info("Reference size %s\tSignal size %s" % (FmriMatr.shape, ICRefMax.shape) )
	FmriMat=FmriMatr - np.tile(np.mean(FmriMatr,1),(m,1)).T
	CovFmri = np.matmul(FmriMat, FmriMat.T) / m
	logger.info("Performing PCA reduction on signal")
	[D,E]=np.linalg.eig(CovFmri)
	EsICnum=ICRefMax.shape[0] #EsICnum can be a number that is less than size(ICRefMax,1)
	index = np.argsort(D)
	eigenvalues = D[index]
	cols=E.shape[1]
	Esort=np.zeros(E.shape)
	dsort=np.zeros(eigenvalues.shape)
	for i in range(cols):
	    Esort[:,i] = E[:,index[cols-i-1] ]
	    dsort[i]   = eigenvalues[index[cols-i-1] ]

	thr=0 #you can change the parameter. such as thr=0.02
	numpc=0
	for i in range(cols):
	    if dsort[i]>thr:
	        numpc=numpc+1
	logger.info("Performing PCA for %d components" % numpc)

	Epart=Esort[:,1:numpc]
	dpart=dsort[1:numpc]
	Lambda_part=np.diag(dpart)
	logger.info("Whitening source signal")
	WhitenMatrix=np.matmul((np.linalg.inv(sqrtm(Lambda_part))), Epart.T)
	logger.info("Done whitening")
	Y=np.matmul(WhitenMatrix, FmriMat)
	logger.info("Done projecting")
	logger.info("Normalizing...")
	if thr<1e-10 and numpc<n:
		for i in range(Y.shape[0]):
			Y[i,:]=Y[i,:]/np.std(Y[i,:])

	logger.info("Normalizing source signal")
	Yinv=np.linalg.pinv(Y)
	ICRefMaxN=np.zeros((EsICnum,m2))
	ICRefMaxC=ICRefMax - np.tile(np.mean(ICRefMax,1), (m2, 1)).T
	for i in range(EsICnum):
	    ICRefMaxN[i,:]=ICRefMaxC[i,:]/np.std(ICRefMaxC[i,:])
	
	logger.info("Computing negentropy")
	NegeEva=np.zeros((EsICnum,1))
	for i in range(EsICnum):
	    NegeEva[i]=nege(ICRefMaxN[i,:])

	iternum=10
	a=0.5
	b=1-a
	EGv=0.3745672075
	ErChuPai=2/np.pi
	ICOutMax=np.zeros((EsICnum,m))
	logger.info("Starting with EGv=%f" % EGv)
	for ICnum in range(EsICnum):
		logger.info('gigicar component: %d/%d' % (ICnum, EsICnum))
		reference=ICRefMaxN[ICnum,:]
		wc=(np.matmul(reference, Yinv)).T
		wc=wc/np.linalg.norm(wc)
		y1=np.matmul(wc.T, Y)
		EyrInitial=(1/m)*(y1)*reference.T
		NegeInitial=nege(y1)
		c=(np.tan((EyrInitial*np.pi)/2))/NegeInitial
		IniObjValue=a*ErChuPai*np.arctan(c*NegeInitial)+b*EyrInitial

		itertime=1
		Nemda=1
		for i in range(iternum):
			Cosy1=np.cosh(y1)
			logCosy1=np.log(Cosy1)
			EGy1=np.mean(logCosy1)
			Negama=EGy1-EGv
			EYgy=(1/m)*Y*(np.tanh(y1)).T
			Jy1=(EGy1-EGv)**2
			KwDaoshu=ErChuPai*c*(1/(1+(c*Jy1)**2))
			Simgrad=(1/m)*Y*reference.T
			g=a*KwDaoshu*2*Negama*EYgy+b*Simgrad
			logging.info(g.shape)
			gtg = np.matmul(g.T, g)
			d=g/(gtg)**0.5
			wx=wc+Nemda*d
			wx=wx/np.linalg.norm(wx)
			y3=wx.T.dot(Y)
			PreObjValue=a*ErChuPai*np.arctan(c*nege(y3))+b*(1/m)*y3*reference.T
			ObjValueChange=PreObjValue-IniObjValue
			ftol=0.02
			dg=g.T*d
			ArmiCondiThr=Nemda*ftol*dg
			if ObjValueChange<ArmiCondiThr:
				Nemda=Nemda/2
				continue
			if (wc-wx).T*(wc-wx) <1.e-5:
				break
			elif itertime==iternum:
				break
			IniObjValue=PreObjValue
			y1=y3
			wc=wx
			itertime=itertime+1
		Source=wx.T*Y
		ICOutMax[ICnum,:]=Source
	TCMax=(1/m)*FmriMatr*ICOutMax.T
	return ICOutMax,TCMax

def nege(x):
	y=np.log(np.cosh(x))
	E1=np.mean(y)
	E2=0.3745672075
	return (E1- E2)**2

def mask_img(img, mask):
	return img[mask==1,:]


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

if len(sys.argv) != 6:
    print("Usage: python gigicar.py sub_id func_file out_dir mask_file template_file ")
    print(sys.argv)
    sys.exit()

sub_id = sys.argv[1]
print(f"sub_id:{sub_id}")

func_file = sys.argv[2]
print(f"func_file:{func_file}")

output_dir = sys.argv[3]
print(f"output_dir:{output_dir}")

mask_file = sys.argv[4]
print(f"mask_file:{mask_file}")

template_file = sys.argv[5]
print(f"template_file:{template_file}")


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




import scipy.io as sio




mask = sio.loadmat('mask.mat')['mask'].flatten()
ref_img = nib.load(template_file)
ref_img = np.array(ref_img.dataobj)
print(f'ref_img.shape {ref_img.shape}')
ref_img = ref_img.reshape(np.prod(ref_img.shape[0:3]), ref_img.shape[3])


print(f'template_file {template_file}')
ref_img = mask_img(ref_img, mask)
src_img = nib.load(func_file)


print(f'func_file {func_file}')
src_img = np.array(src_img.dataobj)
src_img = src_img.reshape(np.prod(src_img.shape[0:3]), src_img.shape[3])
print(f'src_img.shape {src_img.shape}')

src_img = mask_img(src_img, mask)
print(f'src_img.shape {src_img.shape}')
print(f'ref_img.shape {ref_img.shape}')
ICOutMax, TCMax = gigicar(src_img.T, ref_img.T)

