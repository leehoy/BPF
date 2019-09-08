# from Reconstruction import Reconstruction
from Reconstruction_pycuda import Reconstruction
import numpy as np
import glob, sys, os
import logging
import time
import matplotlib.pyplot as plt

pi = np.pi
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)
# data = np.fromfile('Shepp_Logal_3d_256.dat', dtype=np.float32).reshape([256, 256, 256])
nu = 512
nv = 384
du = 400.0 / nu
dv = 300.0 / nv
ns = 360
nx = 256
ny = 256
nz = 256
SAD = 1000.0
SDD = 1500.0
fov = np.sin(np.arctan((nu * du / 2) / SDD)) * SAD
dx = fov * 2.0 / nx
dy = fov * 2.0 / ny
dz = 1.0
params = {'SourceInit': [0, SAD, 0], 'DetectorInit': [0, -(SDD - SAD), 0], 'StartAngle': 0, 'EndAngle': 2 * pi,
          'NumberOfDetectorPixels': [nu, nv], 'DetectorPixelSize': [du, dv], 'NumberOfViews': ns,
          'ImagePixelSpacing': [dx, dy, dz], 'NumberOfImage': [nx, ny, nz], 'PhantomCenter': [0, 0, 0],
          'RotationOrigin': [0, 0, 0], 'ReconCenter': [0, 0, 0], 'Method': 'Distance', 'FilterType': 'hann',
          'cutoff': 1, 'GPU': 1, 'DetectorShape': 'Flat', 'Pitch': 0, 'DetectorOffset': [0, 0]}
R = Reconstruction(params)
filename = 'Shepp_Logan_3d_256.dat'
# filename = 'Shepp_Logan_3d_256.dat'

R.LoadRecon(filename)
ph = R.image
start_time = time.time()
R.forward()
log.info('Forward %f' % (time.time() - start_time))
print(R.proj.shape)
proj0 = np.copy(R.proj)
# R.SaveProj('proj_SheppLogan512_720.dat')
# params = {'SourceInit': [0, SAD, 0], 'DetectorInit': [0, -(SDD - SAD), 0], 'StartAngle': 0, 'EndAngle': 2 * pi,
#           'NumberOfDetectorPixels': [nu, nv], 'DetectorPixelSize': [du, dv], 'NumberOfViews': ns,
#           'ImagePixelSpacing': [dx, dy, dz], 'NumberOfImage': [nx, ny, nz], 'PhantomCenter': [0, 0, 0],
#           'RotationOrigin': [0, 0, 0], 'ReconCenter': [0, 0, 0], 'Method': 'Distance', 'FilterType': 'hann',
#           'cutoff': 1, 'GPU': 1, 'DetectorShape': 'Flat', 'Pitch': 0, 'DetectorOffset': [0, 0]}
# R = Reconstruction(params)
# R.proj = np.copy(proj0)
R.DerivProj()
R.Select_PI()
R.backward_bpf()
R.image.tofile('DBP_SheppLogan256_360_bpf.raw')
R.bpf_constant()
R.bpf_filter()
# dbp.tofile('DBT_SheppLogan256_360_bpf.raw')
R.P0.tofile('Constant.raw', sep='', format='')
R.filtering_weight.tofile('Filter_weight.raw', sep='', format='')
for i in range(R.image.shape[0]):
    P0 = R.P0[i, :, :]
    image = R.image[i, :, :]
    P0[np.where(R.filtering_weight == 0)] = 0
    image[np.where(R.filtering_weight == 0)] = 0
    tmp = (image + P0 / pi) / (R.filtering_weight + 1e-4)
    tmp[np.where(R.filtering_weight == 0)] = 0
    R.image[i, :, :] = np.copy(tmp)
# R.image[np.where(np.isnan(R.image))] = 0
# R.image[np.where(np.isinf(R.image))] = 0
R.SaveRecon('Recon_SheppLogan256_360_bpf.raw')
# plt.imshow(R.proj[1, :, :])
# plt.show()
# start_time = time.time()
# R.Filtering()
# R.backward()
# log.info('Backward: %f' % (time.time() - start_time))
# R.SaveRecon('Recon_SheppLogan512_720_fdk.dat')
