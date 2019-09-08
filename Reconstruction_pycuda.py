import os
import sys
import numpy as np
import numpy.matlib
from scipy.interpolate import interp2d, griddata
import glob
import matplotlib.pyplot as plt
import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy.matlib
import pycuda.gpuarray
from math import ceil
import time
from GPUFuncs_pycuda import *
import logging
from scipy.signal import hilbert
from scipy.interpolate import RegularGridInterpolator

# define logger
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# function alias starts
sin = np.sin
cos = np.cos
atan = np.arctan
tan = np.tan
sinc = np.sinc
sqrt = np.sqrt
repmat = numpy.matlib.repmat
# ceil = np.ceil
pi = np.pi
floor = np.floor
log2 = np.log2
fft = np.fft.fft
ifft = np.fft.ifft
fftshift = np.fft.fftshift
ifftshift = np.fft.ifftshift
real = np.real
# function alias ends
mod = DefineGPUFuns()


class ErrorDescription(object):
    def __init__(self, value):
        if (value == 1):
            self.msg = 'Unknown variables'
        elif (value == 2):
            self.msg = 'Unknown data precision'
        elif (value == 3):
            self.msg = 'Number of file is different from number of projection data required'
        elif (value == 4):
            self.msg = 'Cutoff have to be pose between 0 and 0.5'
        elif (value == 5):
            self.msg = 'Smooth have to be pose between 0 and 1'
        else:
            self.msg = 'Unknown error'

    def __str__(self):
        return self.msg


class Reconstruction(object):

    def __init__(self, params):
        self.params = {'SourceInit': [0, 0, 0], 'DetectorInit': [0, 0, 0], 'StartAngle': 0, 'EndAngle': 0,
                       'NumberOfDetectorPixels': [0, 0], 'DetectorPixelSize': [0, 0], 'NumberOfViews': 0,
                       'ImagePixelSpacing': [0, 0, 0], 'NumberOfImage': [0, 0, 0], 'PhantomCenter': [0, 0, 0],
                       'RotationOrigin': [0, 0, 0], 'Method': 'Distance', 'FilterType': 'ram-lak',
                       'ReconCenter': [0, 0, 0], 'cutoff': 1, 'GPU': 0, 'DetectorShape': 'Flat', 'Pitch': 0,
                       'DetectorOffset': [0, 0]}
        self.params = params
        [self.nu, self.nv] = self.params['NumberOfDetectorPixels']
        [self.du, self.dv] = self.params['DetectorPixelSize']
        [self.nx, self.ny, self.nz] = self.params['NumberOfImage']
        [self.dx, self.dy, self.dz] = self.params['ImagePixelSpacing']
        self.Origin = np.array(self.params['RotationOrigin'])
        self.PhantomCenter = np.array(self.params['PhantomCenter'])
        self.cutoff = self.params['cutoff']
        self.DetectorShape = self.params['DetectorShape']
        self.P = self.params['Pitch']
        self.Source = np.array(self.params['SourceInit'])
        self.Detector = np.array(self.params['DetectorInit'])
        self.SAD = np.sqrt(np.sum((self.Source - self.Origin) ** 2.0))
        self.SDD = np.sqrt(np.sum((self.Source - self.Detector) ** 2.0))
        self.HelicalTrans = self.P * (self.nv * self.dv) * self.SAD / self.SDD
        self.nView = self.params['NumberOfViews']
        self.sAngle = self.params['StartAngle']
        self.eAngle = self.params['EndAngle']
        self.Proj2pi = self.nView / ((self.eAngle - self.sAngle) / (2 * pi))
        self.Method = self.params['Method']
        self.FilterType = self.params['FilterType']
        self.source_z0 = self.Source[2]
        self.detector_z0 = self.Detector[2]
        self.ReconCenter = self.params['ReconCenter']
        self.DetectorOffset = self.params['DetectorOffset']
        if self.params['GPU'] == 1:
            self.GPU = True
        else:
            self.GPU = False

    def LoadProj(self, filename, dtype=np.float32):
        self.proj = np.fromfile(filename, dtype=dtype).reshape([self.nView, self.nv, self.nu])

    def LoadRecon(self, filename, dtype=np.float32):
        self.image = np.fromfile(filename, dtype=dtype).reshape([self.nz, self.ny, self.nx])

    def SaveProj(self, filename):
        self.proj.tofile(filename, sep='', format='')

    def SaveRecon(self, filename):
        self.image.tofile(filename, sep='', format='')

    def Select_PI(self):
        nu = self.nu
        nv = self.nv
        du = self.du
        dv = -1 * self.dv
        nx = self.nx
        ny = self.ny
        nz = self.nz
        dx = self.dx
        dy = -1 * self.dy
        dz = -1 * self.dz
        nViews = self.nView
        R = self.SAD
        D = self.SDD
        sAngle = self.sAngle
        eAngle = self.eAngle
        angle = np.linspace(sAngle, eAngle, nViews + 1)
        angle = angle[0:-1]
        Source = self.Source
        Detector = self.Detector
        source_z0 = self.source_z0
        detector_z0 = self.detector_z0
        H = self.HelicalTrans
        PhantomCenter = self.PhantomCenter
        ReconCenter = self.ReconCenter

        DetectorOffset = self.DetectorOffset

        dtheta = angle[1] - angle[0]
        angle_in = np.zeros([ny, nx], dtype=np.float32)
        angle_out = np.zeros([ny, nx], dtype=np.float32)
        weight = np.zeros([ny, nx], dtype=np.float32)
        x = np.arange(-(nx - 1.0) / 2.0, (nx - 1.0) / 2.0 + 1) * dx
        y = np.arange(-(ny - 1.0) / 2.0, (ny - 1.0) / 2.0 + 1) * dy
        fov = np.sin(np.arctan((nu * du / 2) / D)) * R
        for i in range(ny):
            y1 = y[i]
            r1 = np.arccos(y1 / R)
            angle_in[i, :] = r1
            angle_out[i, :] = -r1 + 2 * pi
            x1 = -sqrt(fov ** 2 - y1 ** 2)
            x2 = sqrt(fov ** 2 - y1 ** 2)
            w1 = x - x1
            w2 = x2 - x
            w1[np.where(w1 < 0)] = 0
            w2[np.where(w2 < 0)] = 0
            weight[i, :] = np.sqrt(w1 * w2)
        self.angle_in = angle_in
        self.angle_out = angle_out
        self.filtering_weight = weight

    def DerivProj(self):
        nu = self.nu
        nv = self.nv
        du = self.du
        dv = -1 * self.dv
        nx = self.nx
        ny = self.ny
        nz = self.nz
        dx = self.dx
        dy = -1 * self.dy
        dz = -1 * self.dz
        nViews = self.nView
        R = self.SAD
        D = self.SDD
        sAngle = self.sAngle
        eAngle = self.eAngle
        angle = np.linspace(sAngle, eAngle, nViews + 1)
        angle = angle[0:-1]
        Source = self.Source
        Detector = self.Detector
        source_z0 = self.source_z0
        detector_z0 = self.detector_z0
        H = self.HelicalTrans
        PhantomCenter = self.PhantomCenter
        ReconCenter = self.ReconCenter

        DetectorOffset = self.DetectorOffset

        dtheta = angle[1] - angle[0]
        u = np.arange(-(nu - 1.0) / 2.0, (nu - 1.0) / 2.0 + 1) * du
        v = np.arange(-(nv - 1.0) / 2.0, (nv - 1.0) / 2.0 + 1) * dv
        [uu, vv] = np.meshgrid(u, v)
        A = np.sqrt(uu ** 2 + vv ** 2 + D ** 2)
        # print(len(u), len(v))
        self.proj_org = np.copy(self.proj)
        proj0 = np.copy(self.proj)
        proj1 = np.append(proj0, np.expand_dims(proj0[0, :, :], axis=0), axis=0)
        dpdu = np.zeros(proj0.shape, dtype=np.float32)
        proj0 = R * proj0 / A
        dpdl = (proj1[1:, :, :] - proj1[:-1, :, :]) / dtheta
        dpdu[:, :, :-1] = (proj0[:, :, 1:] - proj0[:, :, :-1]) / du
        assert dpdl.shape == (nViews, nv, nu) and dpdu.shape == (nViews, nv, nu)
        self.dpdu = dpdu
        self.dpdl = dpdl

    def CurvedDetectorConstruction(self, Source, DetectorCenter, SDD, angle):
        eu = [cos(angle), sin(angle), 0]
        ew = [sin(angle), -cos(angle), 0]
        ev = [0, 0, 1]
        self.da = self.du / self.SDD
        u = (np.arange(0, self.nu) - (self.nu - 1) / 2.0) * self.da
        v = (np.arange(0, self.nv) - (self.nv - 1) / 2.0) * self.dv
        u += self.DetectorOffset[0]
        v += self.DetectorOffset[1]
        DetectorIndex = np.zeros([3, len(v), len(u)], dtype=np.float32)
        U, V = np.meshgrid(u, v)
        # V, U = np.meshgrid(v, u)
        DetectorIndex[0, :, :] = Source[0] + SDD * sin(U) * eu[0] + SDD * cos(U) * ew[0] - V * ev[0]
        DetectorIndex[1, :, :] = Source[1] + SDD * sin(U) * eu[1] + SDD * cos(U) * ew[1] - V * ev[1]
        DetectorIndex[2, :, :] = Source[2] + SDD * sin(U) * eu[2] + SDD * cos(U) * ew[2] - V * ev[2]
        u2 = (np.arange(0, self.nu + 1) - (self.nu - 1) / 2.0) * self.da - self.da / 2.0
        v2 = (np.arange(0, self.nv + 1) - (self.nv - 1) / 2.0) * self.dv - self.dv / 2.0
        u2 += self.DetectorOffset[0]
        v2 += self.DetectorOffset[1]
        DetectorBoundary = np.zeros([3, len(v2), len(u2)], dtype=np.float32)
        U2, V2 = np.meshgrid(u2, v2)
        DetectorBoundary[0, :, :] = Source[0] + SDD * sin(U2) * eu[0] + SDD * cos(U2) * ew[0] - V2 * ev[0]
        DetectorBoundary[1, :, :] = Source[1] + SDD * sin(U2) * eu[1] + SDD * cos(U2) * ew[1] - V2 * ev[1]
        DetectorBoundary[2, :, :] = Source[2] + SDD * sin(U2) * eu[2] + SDD * cos(U2) * ew[2] - V2 * ev[2]
        return DetectorIndex, DetectorBoundary

    def FlatDetectorConstruction(self, Source, DetectorCenter, SDD, angle):
        tol_min = 1e-5
        tol_max = 1e6
        eu = [cos(angle), sin(angle), 0]
        ew = [sin(angle), -cos(angle), 0]
        ev = [0.0, 0.0, 1.0]
        # [nu, nv] = self.params['NumberOfDetectorPixels']
        # [du, dv] = self.params['DetectorPixelSize']
        # [dx, dy, dz] = self.params['ImagePixelSpacing']
        # [nx, ny, nz] = self.params['NumberOfImage']
        # dv=-1.0*dv
        u = (np.arange(0, self.nu) - (self.nu - 1.0) / 2.0) * self.du
        v = (np.arange(0, self.nv) - (self.nv - 1.0) / 2.0) * self.dv
        u += self.DetectorOffset[0]
        v += self.DetectorOffset[1]
        DetectorIndex = np.zeros([3, len(v), len(u)], dtype=np.float32)
        U, V = np.meshgrid(u, v)
        DetectorIndex[0, :, :] = Source[0] + U * eu[0] + SDD * ew[0] - V * ev[0]
        DetectorIndex[1, :, :] = Source[1] + U * eu[1] + SDD * ew[1] - V * ev[1]
        DetectorIndex[2, :, :] = Source[2] + U * eu[2] + SDD * ew[2] - V * ev[2]
        u2 = (np.arange(0, self.nu + 1) - (self.nu - 1) / 2.0) * self.du - self.du / 2.0
        v2 = (np.arange(0, self.nv + 1) - (self.nv - 1) / 2.0) * self.dv - self.dv / 2.0
        u2 += self.DetectorOffset[0]
        v2 += self.DetectorOffset[1]
        DetectorBoundary = np.zeros([3, len(v2), len(u2)], dtype=np.float32)
        U2, V2 = np.meshgrid(u2, v2)
        DetectorBoundary[0, :, :] = Source[0] + U2 * eu[0] + SDD * ew[0] - V2 * ev[0]
        DetectorBoundary[1, :, :] = Source[1] + U2 * eu[1] + SDD * ew[1] - V2 * ev[1]
        DetectorBoundary[2, :, :] = Source[2] + U2 * eu[2] + SDD * ew[2] - V2 * ev[2]
        return DetectorIndex, DetectorBoundary

    @staticmethod
    def _optimalGrid(GridSize):
        if (sqrt(GridSize).is_integer()):
            gridX = int(np.sqrt(GridSize))
            gridY = gridX
        else:
            Candidates = np.arange(1, GridSize + 1)
            Division = GridSize / Candidates
            CheckInteger = Division % 1
            Divisors = Candidates[np.where(CheckInteger == 0)]
            DivisorIndex = int(len(Divisors) / 2)
            gridX = Divisors[DivisorIndex]
            gridY = Divisors[DivisorIndex - 1]
        return (int(gridX), int(gridY))

    @staticmethod
    def Filter(N, pixel_size, FilterType, cutoff):
        '''
        TO DO: Ram-Lak filter implementation
                   Argument for name of filter
        '''
        try:
            if cutoff > 1 or cutoff < 0:
                raise ErrorDescription(4)
        except ErrorDescription as e:
            print(e)
        x = np.arange(0, N) - (N - 1) / 2.0
        h = np.zeros(len(x))
        h[np.where(x == 0)] = 1 / (8 * pixel_size ** 2)
        odds = np.where(x % 2.0 == 1)
        h[odds] = -0.5 / (pi * pixel_size * x[odds]) ** 2
        h = h[0:-1]
        filter = abs(fftshift(fft(h)))
        w = 2 * pi * x[0:-1] / (N - 1)
        # print(filter.shape, w.shape)
        if FilterType == 'ram-lak':
            pass  # Do nothing
        elif FilterType == 'shepp-logan':
            zero = np.where(w == 0)
            tmp = filter[zero]
            filter = filter * sin(w / (2.0 * cutoff)) / (w / (2.0 * cutoff))
            filter[zero] = tmp * sin(w[zero] / (2.0 * cutoff))
        elif FilterType == 'cosine':
            filter = filter * cos(w / (2.0 * cutoff))
        elif FilterType == 'hamming':
            filter = filter * (0.54 + 0.46 * (cos(w / cutoff)))
        elif FilterType == 'hann':
            filter = filter * (0.5 + 0.5 * cos(w / cutoff))

        filter[np.where(abs(w) > pi * cutoff / (2.0 * pixel_size))] = 0
        return filter

    def Filtering(self):
        ki = (np.arange(0, self.nu + 1) - self.nu / 2.0) * self.du
        p = (np.arange(0, self.nv + 1) - self.nv / 2.0) * self.dv
        for i in range(self.proj.shape[0]):
            self.proj[i, :, :] = self.filter_proj(self.proj[i, :, :], ki, p)

    def filter_proj(self, proj, ki, p):
        nu = self.nu
        nv = self.nv
        du = self.du
        dv = -self.dv
        ZeroPaddedLength = int(2 ** (ceil(log2(2.0 * (nu - 1)))))
        R = self.SAD
        D = self.SDD - R
        [kk, pp] = np.meshgrid(ki[0:-1] * R / (R + D), p[0:-1] * R / (R + D))
        weight = R / (sqrt(R ** 2.0 + kk ** 2.0 + pp ** 2.0))

        deltaS = du * R / (R + D)
        filter = Reconstruction.Filter(
            ZeroPaddedLength + 1, du * R / (D + R), self.FilterType, self.cutoff)
        weightd_proj = weight * proj
        Q = np.zeros(weightd_proj.shape, dtype=np.float32)
        for k in range(nv):
            tmp = real(ifft(ifftshift(filter * fftshift(fft(weightd_proj[k, :], ZeroPaddedLength)))))
            Q[k, :] = tmp[0:nu] * deltaS

        return Q

    def bpf_filter(self, N=4096):
        nu = self.nu
        nv = self.nv
        du = self.du
        dv = -1 * self.dv
        nx = self.nx
        ny = self.ny
        nz = self.nz
        dx = self.dx
        dy = -1 * self.dy
        dz = -1 * self.dz
        nViews = self.nView
        R = self.SAD
        D = self.SDD

        x = np.arange(-(nx - 1.0) / 2.0, (nx - 1.0) / 2.0 + 1) * dx
        y = np.arange(-(ny - 1.0) / 2.0, (ny - 1.0) / 2.0 + 1) * dy
        [xx, yy] = np.meshgrid(x, y)
        fov = np.sin(np.arctan((nu * du / 2) / D)) * R
        recon_filt = np.zeros(self.image.shape, dtype=np.float32)
        fov_mask = np.zeros([ny, nx], dtype=np.float32)
        fov_mask[np.sqrt(xx ** 2 + yy ** 2) <= fov] = 1
        for i in range(nz):
            for j in range(ny):
                line = np.where(fov_mask[j, :] > 0)
                # print(line, len(line[0]))
                tmp = -np.imag(hilbert(self.image[i, j, line] * self.filtering_weight[j, line], N)) / (2 * pi)
                # print(tmp.shape)
                recon_filt[i, j, line] = tmp[0, :len(line[0])]
        self.image = recon_filt

    def bpf_constant(self):
        nu = self.nu
        nv = self.nv
        du = self.du
        dv = -1 * self.dv
        nx = self.nx
        ny = self.ny
        nz = self.nz
        dx = self.dx
        dy = -1 * self.dy
        dz = -1 * self.dz
        nViews = self.nView
        R = self.SAD
        D = self.SDD
        u = (np.arange(-(self.nu - 1.0) / 2.0, (self.nu - 1.0) / 2.0 + 1)) * du
        v = (np.arange(-(self.nv - 1.0) / 2.0, (self.nv - 1.0) / 2.0 + 1)) * dv
        assert len(u) == nu and len(v) == nv
        u += self.DetectorOffset[0]
        v += self.DetectorOffset[1]
        sAngle = self.sAngle
        eAngle = self.eAngle
        angle = np.linspace(sAngle, eAngle, nViews + 1)
        angle = angle[0:-1]
        z = np.arange(-(nz - 1.0) / 2.0, (nz - 1.0) / 2.0 + 1) * dz
        ev = np.array([0, 0, 1])
        f = RegularGridInterpolator((angle, -v, u), np.flip(self.proj_org, axis=1), method='linear', bounds_error=False,
                                    fill_value=0.0)
        P0 = np.zeros([nz, ny, nx], dtype=np.float32)
        for i in range(nz):
            for j in range(ny):
                angle1 = self.angle_in[j, 128]
                angle2 = self.angle_out[j, 128]
                r1 = np.array([-R * sin(angle1), R * cos(angle1), 0])
                r2 = np.array([-R * sin(angle2), R * cos(angle2), 0])
                eu1 = np.array([cos(angle1), sin(angle1), 0])
                ew1 = np.array([sin(angle1), -cos(angle1), 0])
                r0 = (r1 + r2) / 2 + ev * z[i]
                u1 = D * np.dot(r0, eu1) / (R + np.dot(r0, ew1))
                v1 = D * z[i] / (R + np.dot(r0, ew1))
                # print(angle1, v1, u1, R, D)
                p1 = f([angle1, v1, u1])
                eu2 = np.array([cos(angle2), sin(angle2), 0])
                ew2 = np.array([sin(angle2), -cos(angle2), 0])
                r0 = (r1 + r2) / 2 + ev * z[i]
                u2 = D * np.dot(r0, eu2) / (R + np.dot(r0, ew2))
                v2 = D * z[i] / (R + np.dot(r0, ew2))
                p2 = f([angle2, v2, u2])
                P0[i, j, :] = (p1 + p2) / 2.0
        self.P0 = P0

    def angular_weight_calculator(self, angle_in, angle_out, dtheta, angle):
        angular_weight = np.zeros(angle_in.shape, dtype=np.float32)
        d_in = (angle - angle_in) / dtheta
        d_out = (angle_out - angle) / dtheta
        ind1 = np.where((angle - dtheta > angle_in) & (angle + dtheta < angle_out))
        ind2 = np.where(angle + dtheta < angle_in)
        ind3 = np.where(angle - dtheta > angle_out)
        ind4 = np.where((angle + dtheta > angle_in) & (angle < angle_in))
        ind5 = np.where((angle > angle_in) & (angle - dtheta < angle_in))
        ind6 = np.where((angle + dtheta > angle_out) & (angle < angle_out))
        ind7 = np.where((angle > angle_out) & (angle - dtheta < angle_out))
        angular_weight[ind1] = 1.0
        angular_weight[ind2] = 0.0
        angular_weight[ind3] = 0.0
        angular_weight[ind4] = ((1 + d_in[ind4]) ** 2) / 2.0
        angular_weight[ind5] = 0.5 + d_in[ind5] - 0.5 * (d_in[ind5]) ** 2
        angular_weight[ind6] = 0.5 + d_out[ind6] - 0.5 * (d_out[ind6]) ** 2
        angular_weight[ind7] = ((1 + d_out[ind7]) ** 2) / 2.0
        angular_weight2 = np.zeros([self.nz, self.ny, self.nx], dtype=np.float32)
        for i in range(self.nz):
            angular_weight2[i, :, :] = angular_weight
        return angular_weight2

    def backward_legacy(self):
        nViews = self.params['NumberOfViews']
        [nu, nv] = self.params['NumberOfDetectorPixels']
        [du, dv] = self.params['DetectorPixelSize']
        [dx, dy, dz] = self.params['ImagePixelSpacing']
        [nx, ny, nz] = self.params['NumberOfImage']
        cutoff = self.params['cutoff']
        FilterType = self.params['FilterType']
        dy = -1 * dy
        dz = -1 * dz
        dv = -1 * dv
        Source_Init = np.array(self.params['SourceInit'])
        Detector_Init = np.array(self.params['DetectorInit'])
        StartAngle = self.params['StartAngle']
        EndAngle = self.params['EndAngle']
        Origin = np.array(self.params['Origin'])
        PhantomCenter = np.array(self.params['PhantomCenter'])
        gpu = self.params['GPU']
        SAD = np.sqrt(np.sum((Source_Init - Origin) ** 2.0))
        SDD = np.sqrt(np.sum((Source_Init - Detector_Init) ** 2.0))
        # Calculates detector center
        # angle = np.linspace(StartAngle, EndAngle, nViews + 1)
        # angle = angle[0:-1]
        # dtheta = angle[1] - angle[0]
        # deltaS = du * SAD / SDD
        # Xplane = (PhantomCenter[0] - nx / 2 + range(0, nx + 1)) * dx
        # Yplane = (PhantomCenter[1] - ny / 2 + range(0, ny + 1)) * dy
        # Zplane = (PhantomCenter[2] - nz / 2 + range(0, nz + 1)) * dz
        # Xpixel = Xplane[0:-1]
        # Ypixel = Yplane[0:-1]
        # Zpixel = Zplane[0:-1]
        # ki = (np.arange(0, nu + 1) - nu / 2.0) * du
        # p = (np.arange(0, nv + 1) - nv / 2.0) * dv
        alpha = 0
        beta = 0
        gamma = 0
        eu = [cos(gamma) * cos(alpha), sin(alpha), sin(gamma)]
        ev = [cos(gamma) * -sin(alpha), cos(gamma) * cos(alpha), sin(gamma)]
        ew = [0, 0, 1]
        # print('Variable initialization: ' + str(time.time() - start_time))
        # Source = np.array([-SAD * sin(angle[0]), SAD * cos(angle[0]), 0])  # z-direction rotation
        # Detector = np.array([(SDD - SAD) * sin(angle[0]), -(SDD - SAD) * cos(angle[0]), 0])
        # DetectorLength = np.array(
        #    [np.arange(floor(-nu / 2), floor(nu / 2) + 1) * du, np.arange(floor(-nv / 2), floor(nv / 2) + 1) * dv])
        # DetectorVectors = [eu, ev, ew]
        # DetectorIndex = self.DetectorConstruction(Detector, DetectorLength, DetectorVectors, angle[0])
        recon = np.zeros([nz, ny, nx], dtype=np.float32)
        # plt.plot(ki)
        # plt.show()
        start_time = time.time()
        if (self.params['Method'] == 'Distance'):
            recon = self.distance_backproj()
        elif (self.params['Method'] == 'Ray'):
            pass
            # recon = self.ray(DetectorIndex, Source, Detector, angle)
        self.image = recon

    def backward(self):
        recon = np.zeros([self.nz, self.ny, self.nx], dtype=np.float32)
        start_time = time.time()
        if self.Method == 'Distance':
            recon = self.distance_backproj()
        elif self.Method == 'Ray':
            pass
            # recon = self.ray_backproj(DetectorIndex, Source, Detector, angle)
        self.image = recon

    def backward_bpf(self):
        nu = self.nu
        nv = self.nv
        du = self.du
        dv = -1 * self.dv
        nx = self.nx
        ny = self.ny
        nz = self.nz
        dx = self.dx
        dy = -1 * self.dy
        dz = -1 * self.dz
        nViews = self.nView
        R = self.SAD
        D = self.SDD
        sAngle = self.sAngle
        eAngle = self.eAngle
        angle = np.linspace(sAngle, eAngle, nViews + 1)
        angle = angle[0:-1]
        Source = self.Source
        Detector = self.Detector
        source_z0 = self.source_z0
        detector_z0 = self.detector_z0
        H = self.HelicalTrans
        PhantomCenter = self.PhantomCenter
        ReconCenter = self.ReconCenter

        DetectorOffset = self.DetectorOffset

        dtheta = angle[1] - angle[0]
        Xpixel = ReconCenter[0] + (np.arange(0, nx) - (nx - 1) / 2.0) * dx
        Ypixel = ReconCenter[1] + (np.arange(0, ny) - (ny - 1) / 2.0) * dy
        Zpixel = ReconCenter[2] + (np.arange(0, nz) - (nz - 1) / 2.0) * dz
        ki = (np.arange(0, nu + 1) - (nu - 1) / 2.0) * du
        p = (np.arange(0, nv + 1) - (nv - 1) / 2.0) * dv
        ki += DetectorOffset[0]
        p += DetectorOffset[1]
        recon = np.zeros([nz, ny, nx], dtype=np.float32)
        if self.GPU:
            device = drv.Device(0)
            attrs = device.get_attributes()
            MAX_THREAD_PER_BLOCK = attrs[pycuda._driver.device_attribute.MAX_THREADS_PER_BLOCK]
            MAX_GRID_DIM_X = attrs[pycuda._driver.device_attribute.MAX_GRID_DIM_X]
            TotalSize = nx * ny * nz
            if (TotalSize < MAX_THREAD_PER_BLOCK):
                blockX = nx * ny * nz
                blockY = 1
                blockZ = 1
                gridX = 1
                gridY = 1
            else:
                blockX = 16
                blockY = 16
                blockZ = 1
                GridSize = ceil(TotalSize / (blockX * blockY))
                try:
                    if (GridSize < MAX_GRID_DIM_X):
                        [gridX, gridY] = Reconstruction._optimalGrid(GridSize)
                    else:
                        raise ErrorDescription(6)
                except ErrorDescription as e:
                    print(e)
                    sys.exit()
            try:
                if self.DetectorShape == 'Curved':
                    du = du / D
                    distance_backproj_arb = mod.get_function('curved_distance_backproj_arb')
                elif self.DetectorShape == 'Flat':
                    # distance_backproj_arb = mod.get_function('flat_distance_backproj_arb')
                    # distance_backproj_arb = mod.get_function('flat_distance_backproj_arb2')
                    distance_backproj_arb = mod.get_function('flat_distance_backproj_arb_pi')
                else:
                    raise ErrorDescription(1)
            except ErrorDescription as e:
                print(e)
                sys.exit()

            dest = pycuda.gpuarray.to_gpu(recon.flatten().astype(np.float32))
            angular_weight_gpu = pycuda.gpuarray.to_gpu(recon.flatten().astype(np.float32))
            x_pixel_gpu = pycuda.gpuarray.to_gpu(Xpixel.astype(np.float32))
            y_pixel_gpu = pycuda.gpuarray.to_gpu(Ypixel.astype(np.float32))
            z_pixel_gpu = pycuda.gpuarray.to_gpu(Zpixel.astype(np.float32))
            u_plane_gpu = pycuda.gpuarray.to_gpu(ki.astype(np.float32))
            v_plane_gpu = pycuda.gpuarray.to_gpu(p.astype(np.float32))
            recon_param = np.array(
                [dx, dy, dz, nx, ny, nz, nu, nv, du, dv, Source[0], Source[1], Source[2], Detector[0], Detector[1],
                 Detector[2], angle[0], 0.0, R, 0]).astype(np.float32)
            recon_param_gpu = pycuda.gpuarray.zeros(20, np.float32)
            # recon_param_gpu = drv.mem_alloc(recon_param.nbytes)
            recon_param_gpu = pycuda.gpuarray.to_gpu(recon_param.flatten().astype(np.float32))
            # Q = self.proj * dtheta
            # Q_gpu = pycuda.gpuarray.to_gpu(Q.flatten().astype(np.float32))
            for i in range(nViews):
                log.debug(i)
                Q1 = self.dpdu[i, :, :] * dtheta
                Q1 = Q1.flatten().astype(np.float32)
                Q2 = self.dpdl[i, :, :] * dtheta
                Q2 = Q2.flatten().astype(np.float32)
                Source[2] = source_z0 + H * angle[i] / (2 * pi)
                Detector[2] = detector_z0 + H * angle[i] / (2 * pi)
                angle1 = np.float32(angle[i])
                angle2 = np.float32(0.0)
                angular_weight = self.angular_weight_calculator(self.angle_in, self.angle_out, dtheta, angle1)
                # angular_weight_gpu = pycuda.gpuarray.to_gpu(Q.flatten().astype(np.float32))
                angular_weight_gpu.set(angular_weight.flatten().astype(np.float32))
                recon_param = np.array(
                    [dx, dy, dz, nx, ny, nz, nu, nv, du, dv, Source[0], Source[1], Source[2], Detector[0], Detector[1],
                     Detector[2], angle[i], 0.0, R, D]).astype(np.float32)
                recon_param_gpu.set(recon_param.flatten())
                # print(blockX, blockY, blockZ, gridX, gridY)
                distance_backproj_arb(dest, drv.In(Q1), drv.In(Q2), x_pixel_gpu, y_pixel_gpu, z_pixel_gpu, u_plane_gpu,
                                      v_plane_gpu, angular_weight_gpu, recon_param_gpu, block=(blockX, blockY, blockZ),
                                      grid=(gridX, gridY))
                # pycuda.autoinit.context.synchronize()
            del u_plane_gpu, v_plane_gpu, x_pixel_gpu, y_pixel_gpu, z_pixel_gpu, recon_param_gpu
            recon = dest.get().reshape([nz, ny, nx]).astype(np.float32)
            del dest
        else:
            for i in range(nViews):
                Q = self.proj[i, :, :]
                recon += self._distance_backproj_arb(Q, Xpixel, Ypixel, Zpixel, ki, p, angle[i], 0.0,
                                                     self.params) * dtheta
        self.image = recon

    def distance_backproj(self):
        # proj should be filtered data
        nu = self.nu
        nv = self.nv
        du = self.du
        dv = -1 * self.dv
        nx = self.nx
        ny = self.ny
        nz = self.nz
        dx = self.dx
        dy = -1 * self.dy
        dz = -1 * self.dz
        nViews = self.nView
        R = self.SAD
        D = self.SDD
        sAngle = self.sAngle
        eAngle = self.eAngle
        angle = np.linspace(sAngle, eAngle, nViews + 1)
        angle = angle[0:-1]
        Source = self.Source
        Detector = self.Detector
        source_z0 = self.source_z0
        detector_z0 = self.detector_z0
        H = self.HelicalTrans
        PhantomCenter = self.PhantomCenter
        ReconCenter = self.ReconCenter

        DetectorOffset = self.DetectorOffset

        dtheta = angle[1] - angle[0]
        Xpixel = ReconCenter[0] + (np.arange(0, nx) - (nx - 1) / 2.0) * dx
        Ypixel = ReconCenter[1] + (np.arange(0, ny) - (ny - 1) / 2.0) * dy
        Zpixel = ReconCenter[2] + (np.arange(0, nz) - (nz - 1) / 2.0) * dz
        ki = (np.arange(0, nu + 1) - (nu - 1) / 2.0) * du
        p = (np.arange(0, nv + 1) - (nv - 1) / 2.0) * dv
        ki += DetectorOffset[0]
        p += DetectorOffset[1]
        recon = np.zeros([nz, ny, nx], dtype=np.float32)
        if self.GPU:
            device = drv.Device(0)
            attrs = device.get_attributes()
            MAX_THREAD_PER_BLOCK = attrs[pycuda._driver.device_attribute.MAX_THREADS_PER_BLOCK]
            MAX_GRID_DIM_X = attrs[pycuda._driver.device_attribute.MAX_GRID_DIM_X]
            TotalSize = nx * ny * nz
            if (TotalSize < MAX_THREAD_PER_BLOCK):
                blockX = nx * ny * nz
                blockY = 1
                blockZ = 1
                gridX = 1
                gridY = 1
            else:
                blockX = 32
                blockY = 32
                blockZ = 1
                GridSize = ceil(TotalSize / (blockX * blockY))
                try:
                    if (GridSize < MAX_GRID_DIM_X):
                        [gridX, gridY] = Reconstruction._optimalGrid(GridSize)
                    else:
                        raise ErrorDescription(6)
                except ErrorDescription as e:
                    print(e)
                    sys.exit()
            try:
                if self.DetectorShape == 'Curved':
                    du = du / D
                    distance_backproj_arb = mod.get_function('curved_distance_backproj_arb')
                elif self.DetectorShape == 'Flat':
                    # distance_backproj_arb = mod.get_function('flat_distance_backproj_arb')
                    distance_backproj_arb = mod.get_function('flat_distance_backproj_arb2')
                else:
                    raise ErrorDescription(1)
            except ErrorDescription as e:
                print(e)
                sys.exit()

            dest = pycuda.gpuarray.to_gpu(recon.flatten().astype(np.float32))
            x_pixel_gpu = pycuda.gpuarray.to_gpu(Xpixel.astype(np.float32))
            y_pixel_gpu = pycuda.gpuarray.to_gpu(Ypixel.astype(np.float32))
            z_pixel_gpu = pycuda.gpuarray.to_gpu(Zpixel.astype(np.float32))
            u_plane_gpu = pycuda.gpuarray.to_gpu(ki.astype(np.float32))
            v_plane_gpu = pycuda.gpuarray.to_gpu(p.astype(np.float32))
            recon_param = np.array(
                [dx, dy, dz, nx, ny, nz, nu, nv, du, dv, Source[0], Source[1], Source[2], Detector[0], Detector[1],
                 Detector[2], angle[0], 0.0, R, 0]).astype(np.float32)
            recon_param_gpu = pycuda.gpuarray.zeros(20, np.float32)
            # recon_param_gpu = drv.mem_alloc(recon_param.nbytes)
            recon_param_gpu = pycuda.gpuarray.to_gpu(recon_param.flatten().astype(np.float32))
            Q = self.proj * dtheta
            Q_gpu = pycuda.gpuarray.to_gpu(Q.flatten().astype(np.float32))
            for i in range(nViews):
                log.debug(i)
                # Q = self.proj[i, :, :] * dtheta
                # Q = Q.flatten().astype(np.float32)
                Source[2] = source_z0 + H * angle[i] / (2 * pi)
                Detector[2] = detector_z0 + H * angle[i] / (2 * pi)
                angle1 = np.float32(angle[i])
                angle2 = np.float32(0.0)
                recon_param = np.array(
                    [dx, dy, dz, nx, ny, nz, nu, nv, du, dv, Source[0], Source[1], Source[2], Detector[0], Detector[1],
                     Detector[2], angle[i], 0.0, R, i]).astype(np.float32)
                recon_param_gpu.set(recon_param.flatten())
                # drv.memcpy_htod(recon_param_gpu, recon_param)
                distance_backproj_arb(dest, Q_gpu, x_pixel_gpu, y_pixel_gpu, z_pixel_gpu, u_plane_gpu, v_plane_gpu,
                                      recon_param_gpu, block=(blockX, blockY, blockZ),
                                      grid=(gridX, gridY))
                # distance_backproj_arb(dest, Q_gpu, x_pixel_gpu, y_pixel_gpu, z_pixel_gpu, u_plane_gpu, v_plane_gpu,
                #                      recon_param_gpu, Source[0], Source[1], Source[2], Detector[0], Detector[1],
                #                      Detector[2], angle1, angle2, np.uint16(i), block=(blockX, blockY, blockZ),
                #                      grid=(gridX, gridY))
                # pycuda.autoinit.context.synchronize()
            del u_plane_gpu, v_plane_gpu, x_pixel_gpu, y_pixel_gpu, z_pixel_gpu, recon_param_gpu
            recon = dest.get().reshape([nz, ny, nx]).astype(np.float32)
            del dest
        else:
            for i in range(nViews):
                Q = self.proj[i, :, :]
                recon += self._distance_backproj_arb(Q, Xpixel, Ypixel, Zpixel, ki, p, angle[i], 0.0,
                                                     self.params) * dtheta
        return recon

    @staticmethod
    def _distance_backproj_arb(proj, Xpixel, Ypixel, Zpixel, Uplane, Vplane, angle1, angle2, params):
        tol_min = 1e-6
        [nu, nv] = params['NumberOfDetectorPixels']
        [du, dv] = params['DetectorPixelSize']
        [dx, dy, dz] = params['ImagePixelSpacing']
        [nx, ny, nz] = params['NumberOfImage']
        dx = -1 * dx
        dy = -1 * dy
        dv = -1 * dv
        # angle1: rotation angle between point and X-axis
        # angle2: rotation angle between point and XY-palne
        Source = np.array(params['SourceInit'])
        Detector = np.array(params['DetectorInit'])
        R = sqrt(np.sum((np.array(Source) - np.array(params['PhantomCenter'])) ** 2.0))
        recon_pixelsX = Xpixel
        recon_pixelsY = Ypixel
        recon_pixelsZ = Zpixel
        recon = np.zeros([nz, ny, nx], dtype=np.float32)
        f_angle = lambda x, y: atan(x / y) if y != 0 else atan(0) if x == 0 else -pi / 2 if x < 0 else pi / 2
        fx = lambda x, y, z: x * cos(angle2) * cos(angle1) + y * cos(angle2) * sin(angle1) - z * sin(angle2) * cos(
            angle1) * sin(f_angle(x, y)) - z * sin(angle2) * sin(angle1) * cos(f_angle(x, y))
        fy = lambda x, y, z: y * cos(angle2) * cos(angle1) - x * cos(angle2) * sin(angle1) - z * sin(angle2) * cos(
            angle1) * cos(f_angle(x, y)) + z * sin(angle2) * sin(angle1) * sin(f_angle(x, y))
        fz = lambda x, y, z: z * cos(angle2) + sqrt(x ** 2 + y ** 2) * sin(angle2)
        for i in range(nz):
            for j in range(ny):
                for k in range(nx):
                    #                     l = sqrt(recon_pixelsX[k] ** 2 + recon_pixelsY[j] ** 2 + recon_pixelsZ[i] ** 2)
                    xc = fx(recon_pixelsX[k], recon_pixelsY[j], recon_pixelsZ[i])
                    yc = fy(recon_pixelsX[k], recon_pixelsY[j], recon_pixelsZ[i])
                    zc = fz(recon_pixelsX[k], recon_pixelsY[j], recon_pixelsZ[i])
                    #                     yc = -(recon_pixelsX[k]) * sin(angle) + (recon_pixelsY[j]) * cos(angle)
                    x1 = fx((recon_pixelsX[k] - dx / 2), (recon_pixelsY[j] - dy / 2), recon_pixelsZ[i] - dz / 2)
                    y1 = fy((recon_pixelsX[k] - dx / 2), (recon_pixelsY[j] - dy / 2), recon_pixelsZ[i] - dz / 2)
                    z1 = fz((recon_pixelsX[k] - dx / 2), (recon_pixelsY[j] - dy / 2), recon_pixelsZ[i] - dz / 2)

                    x2 = fx((recon_pixelsX[k] + dx / 2), (recon_pixelsY[j] - dy / 2), recon_pixelsZ[i] - dz / 2)
                    y2 = fy((recon_pixelsX[k] + dx / 2), (recon_pixelsY[j] - dy / 2), recon_pixelsZ[i] - dz / 2)
                    z2 = fz((recon_pixelsX[k] + dx / 2), (recon_pixelsY[j] - dy / 2), recon_pixelsZ[i] - dz / 2)

                    x3 = fx((recon_pixelsX[k] + dx / 2), (recon_pixelsY[j] + dy / 2), recon_pixelsZ[i] - dz / 2)
                    y3 = fy((recon_pixelsX[k] + dx / 2), (recon_pixelsY[j] + dy / 2), recon_pixelsZ[i] - dz / 2)
                    z3 = fz((recon_pixelsX[k] + dx / 2), (recon_pixelsY[j] + dy / 2), recon_pixelsZ[i] - dz / 2)

                    x4 = fx((recon_pixelsX[k] - dx / 2), (recon_pixelsY[j] + dy / 2), recon_pixelsZ[i] - dz / 2)
                    y4 = fy((recon_pixelsX[k] - dx / 2), (recon_pixelsY[j] + dy / 2), recon_pixelsZ[i] - dz / 2)
                    z4 = fz((recon_pixelsX[k] - dx / 2), (recon_pixelsY[j] + dy / 2), recon_pixelsZ[i] - dz / 2)

                    x5 = fx((recon_pixelsX[k] - dx / 2), (recon_pixelsY[j] - dy / 2), recon_pixelsZ[i] + dz / 2)
                    y5 = fy((recon_pixelsX[k] - dx / 2), (recon_pixelsY[j] - dy / 2), recon_pixelsZ[i] + dz / 2)
                    z5 = fz((recon_pixelsX[k] - dx / 2), (recon_pixelsY[j] - dy / 2), recon_pixelsZ[i] + dz / 2)

                    x6 = fx((recon_pixelsX[k] + dx / 2), (recon_pixelsY[j] - dy / 2), recon_pixelsZ[i] + dz / 2)
                    y6 = fy((recon_pixelsX[k] + dx / 2), (recon_pixelsY[j] - dy / 2), recon_pixelsZ[i] + dz / 2)
                    z6 = fz((recon_pixelsX[k] + dx / 2), (recon_pixelsY[j] - dy / 2), recon_pixelsZ[i] + dz / 2)

                    x7 = fx((recon_pixelsX[k] + dx / 2), (recon_pixelsY[j] + dy / 2), recon_pixelsZ[i] + dz / 2)
                    y7 = fy((recon_pixelsX[k] + dx / 2), (recon_pixelsY[j] + dy / 2), recon_pixelsZ[i] + dz / 2)
                    z7 = fz((recon_pixelsX[k] + dx / 2), (recon_pixelsY[j] + dy / 2), recon_pixelsZ[i] + dz / 2)

                    x8 = fx((recon_pixelsX[k] - dx / 2), (recon_pixelsY[j] + dy / 2), recon_pixelsZ[i] + dz / 2)
                    y8 = fy((recon_pixelsX[k] - dx / 2), (recon_pixelsY[j] + dy / 2), recon_pixelsZ[i] + dz / 2)
                    z8 = fz((recon_pixelsX[k] - dx / 2), (recon_pixelsY[j] + dy / 2), recon_pixelsZ[i] + dz / 2)

                    slope_u1 = (Source[0] - x1) / (Source[1] - y1)
                    slope_u2 = (Source[0] - x2) / (Source[1] - y2)
                    slope_u3 = (Source[0] - x3) / (Source[1] - y3)
                    slope_u4 = (Source[0] - x4) / (Source[1] - y4)
                    slope_u5 = (Source[0] - x5) / (Source[1] - y5)
                    slope_u6 = (Source[0] - x6) / (Source[1] - y6)
                    slope_u7 = (Source[0] - x7) / (Source[1] - y7)
                    slope_u8 = (Source[0] - x8) / (Source[1] - y8)
                    slopes_u = [slope_u1, slope_u2, slope_u3, slope_u4, slope_u5, slope_u6, slope_u7, slope_u8]
                    slope_l = min(slopes_u)
                    slope_r = max(slopes_u)
                    coord_u1 = (slope_l * Detector[1]) + (Source[0] - slope_r * Source[1])
                    coord_u2 = (slope_r * Detector[1]) + (Source[0] - slope_r * Source[1])
                    u_l = floor((coord_u1 - Uplane[0]) / du)
                    u_r = floor((coord_u2 - Uplane[0]) / du)
                    s_index_u = int(min(u_l, u_r))
                    e_index_u = int(max(u_l, u_r))

                    slope_v1 = (Source[2] - z1) / (Source[1] - y1)
                    slope_v2 = (Source[2] - z2) / (Source[1] - y2)
                    slope_v3 = (Source[2] - z3) / (Source[1] - y3)
                    slope_v4 = (Source[2] - z4) / (Source[1] - y4)
                    slope_v5 = (Source[2] - z5) / (Source[1] - y5)
                    slope_v6 = (Source[2] - z6) / (Source[1] - y6)
                    slope_v7 = (Source[2] - z7) / (Source[1] - y7)
                    slope_v8 = (Source[2] - z8) / (Source[1] - y8)
                    slopes_v = [slope_v1, slope_v2, slope_v3, slope_v4, slope_v5, slope_v6, slope_v7, slope_v8]
                    slope_t = min(slopes_v)
                    slope_b = max(slopes_v)
                    coord_v1 = (slope_t * Detector[2]) + (Source[2] - slope_t * Source[1])
                    coord_v2 = (slope_b * Detector[2]) + (Source[2] - slope_b * Source[1])
                    v_l = floor((coord_v1 - Vplane[0]) / dv)
                    v_r = floor((coord_v2 - Vplane[0]) / dv)
                    s_index_v = int(min(v_l, v_r))
                    e_index_v = int(min(v_l, v_r))
                    for l in range(s_index_v, e_index_v + 1):
                        if (l < 0 or l > nu):
                            continue
                        if (s_index_v == e_index_v):
                            weight1 = 1.0
                        elif (l == s_index_v):
                            weight1 = (max(coord_v1, coord_v2) - Vplane[l + 1]) / abs(coord_v1 - coord_v2)
                        elif (l == e_index_v):
                            weight1 = (Vplane[l] - min(coord_v1, coord_v2)) / abs(coord_v1 - coord_v2)
                        else:
                            weight1 = abs(dv) / abs(coord_v1 - coord_v2)
                        for m in range(s_index_u, e_index_u + 1):
                            if (m < 0 or m > nv):
                                continue
                            if (s_index_u == e_index_u):
                                weight2 = 1.0
                            elif (m == s_index_u):
                                weight2 = (Uplane[k + 1] - min(coord_u1, coord_u2)) / abs(coord_u1 - coord_u2)
                            elif (m == e_index_u):
                                weight2 = (max(coord_u1, coord_u2) - Uplane[k]) / abs(coord_u1 - coord_u2)
                            else:
                                weight2 = abs(du) / abs(coord_u1 - coord_u2)
                            recon[i][j][k] += proj[l][m] * weight1 * weight2 * (R ** 2) / (R - yc) ** 2
        return recon

    @staticmethod
    def _distance_backproj_about_z(proj, Xpixel, Ypixel, Zpixel, Uplane, Vplane, angle, params):
        tol_min = 1e-6
        [nu, nv] = params['NumberOfDetectorPixels']
        [du, dv] = params['DetectorPixelSize']
        [dx, dy, dz] = params['ImagePixelSpacing']
        [nx, ny, nz] = params['NumberOfImage']
        dx = -1 * dx
        dy = -1 * dy
        dv = -1 * dv
        #         SAD = params['SAD']
        #         SDD = parasm['SDD']
        Source = np.array(params['SourceInit'])
        Detector = np.array(params['DetectorInit'])
        R = sqrt(np.sum((np.array(Source) - np.array(params['PhantomCenter'])) ** 2))
        recon_pixelsX = Xpixel
        recon_pixelsY = Ypixel
        recon_pixelsZ = Zpixel
        recon = np.zeros([nz, ny, nx], dtype=np.float32)
        #         recon_pixelsX = Xplane[0:-1] + dx / 2
        #         recon_pixelsY = Yplane[0:-1] + dy / 2
        #         recon_pixelsZ = Zplane[0:-1] + dz / 2
        #         [reconY, reconX] = np.meshgrid(recon_pixelsY, recon_pixlesZ)
        #         reconX_c1 = (reconX + dx / 2) * cos(angle) + (reconY + dy / 2) * sin(angle)
        #         reconX_c2 = (reconX - dx / 2) * cos(angle) + (reconY - dy / 2) * sin(angle)
        #         reconX_c3 = (reconX + dx / 2) * cos(angle) + (reconY - dy / 2) * sin(angle)
        #         reconX_c4 = (reconX - dx / 2) * cos(angle) + (reconY + dy / 2) * sin(angle)
        #
        #         reconY_c1 = -(reconX + dx / 2) * sin(angle) + (reconY + dy / 2) * cos(angle)
        #         reconY_c2 = -(reconX - dx / 2) * sin(angle) + (reconY - dy / 2) * cos(angle)
        #         reconY_c3 = -(reconX + dx / 2) * sin(angle) + (reconY - dy / 2) * cos(angle)
        #         reconY_c4 = -(reconX - dx / 2) * sin(angle) + (reconY + dy / 2) * cos(angle)
        #
        #         SlopeU_c1 = (Source[0] - reconX_c1) / (Source[1] - reconY_c1)
        #         SlopeU_c2 = (Source[0] - reconX_c2) / (Source[1] - reconY_c2)
        #         SlopeU_c3 = (Source[0] - reconX_c3) / (Source[1] - reconY_c3)
        #         SlopeU_c4 = (Source[0] - reconX_c4) / (Source[1] - reconY_c4)
        #         [reconZ, reconY] = np.meshgrid
        for i in range(nz):
            for j in range(ny):
                for k in range(nx):
                    yc = -(recon_pixelsX[k]) * sin(angle) + (recon_pixelsY[j]) * cos(angle)
                    x1 = (recon_pixelsX[k] + dx / 2) * cos(angle) + (recon_pixelsY[j] + dy / 2) * sin(angle)
                    y1 = -(recon_pixelsX[k] + dx / 2) * sin(angle) + (recon_pixelsY[j] + dy / 2) * cos(angle)
                    slope1 = (Source[0] - x1) / (Source[1] - y1)
                    x2 = (recon_pixelsX[k] - dx / 2) * cos(angle) + (recon_pixelsY[j] - dy / 2) * sin(angle)
                    y2 = -(recon_pixelsX[k] - dx / 2) * sin(angle) + (recon_pixelsY[j] - dy / 2) * cos(angle)
                    slope2 = (Source[0] - x2) / (Source[1] - y2)
                    x3 = (recon_pixelsX[k] + dx / 2) * cos(angle) + (recon_pixelsY[j] - dy / 2) * sin(angle)
                    y3 = -(recon_pixelsX[k] + dx / 2) * sin(angle) + (recon_pixelsY[j] - dy / 2) * cos(angle)
                    slope3 = (Source[0] - x3) / (Source[1] - y3)
                    x4 = (recon_pixelsX[k] - dx / 2) * cos(angle) + (recon_pixelsY[j] + dy / 2) * sin(angle)
                    y4 = -(recon_pixelsX[k] - dx / 2) * sin(angle) + (recon_pixelsY[j] + dy / 2) * cos(angle)
                    slope4 = (Source[0] - x4) / (Source[1] - y4)
                    slopes_u = [slope1, slope2, slope3, slope4]
                    slope_l = min(slopes_u)
                    slope_r = max(slopes_u)
                    coord_u1 = (slope_l * Detector[1]) + (Source[0] - slope_r * Source[1])
                    coord_u2 = (slope_r * Detector[1]) + (Source[0] - slope_r * Source[1])
                    u_l = floor((coord_u1 - Uplane[0]) / du)
                    u_r = floor((coord_u2 - Uplane[0]) / du)
                    s_index_u = int(min(u_l, u_r))
                    e_index_u = int(max(u_l, u_r))

                    z1 = recon_pixelsZ[i] - dz / 2
                    z2 = recon_pixelsZ[i] + dz / 2
                    slopes_v = [(Source[2] - z1) / (Source[1] - yc), (Source[2] - z2) / (Source[1] - yc)]
                    slope_t = min(slopes_v)
                    slope_b = max(slopes_v)
                    coord_v1 = (slope_t * Detector[2]) + (Source[2] - slope_t * Source[1])
                    coord_v2 = (slope_b * Detector[2]) + (Source[2] - slope_b * Source[1])
                    v_l = floor((coord_v1 - Vplane[0]) / dv)
                    v_r = floor((coord_v2 - Vplane[0]) / dv)
                    s_index_v = int(min(v_l, v_r))
                    e_index_v = int(min(v_l, v_r))
                    for l in range(s_index_v, e_index_v + 1):
                        if (s_index_v == e_index_v):
                            weight1 = 1.0
                        elif (l == s_index_v):
                            weight1 = (max(coord_v1, coord_v2) - Vplane[l + 1]) / abs(coord_v1 - coord_v2)
                        elif (l == e_index_v):
                            weight1 = (Vplane[l] - min(coord_v1, coord_v2)) / abs(coord_v1 - coord_v2)
                        else:
                            weight1 = abs(dv) / abs(coord_v1 - coord_v2)
                        for m in range(s_index_u, e_index_u + 1):
                            if (s_index_u == e_index_u):
                                weight2 = 1.0
                            elif (m == s_index_u):
                                weight2 = (Uplane[k + 1] - min(coord_u1, coord_u2)) / abs(coord_u1 - coord_u2)
                            elif (m == e_index_u):
                                weight2 = (max(coord_u1, coord_u2) - Uplane[k]) / abs(coord_u1 - coord_u2)
                            else:
                                weight2 = abs(du) / abs(coord_u1 - coord_u2)
                            recon[i][j][k] += proj[l][m] * weight1 * weight2 * (R ** 2) / (R - yc) ** 2
        return recon

    def forward_legacy(self):
        start_time = time.time()

        nViews = self.params['NumberOfViews']
        [nu, nv] = self.params['NumberOfDetectorPixels']
        [du, dv] = self.params['DetectorPixelSize']
        [dx, dy, dz] = self.params['ImagePixelSpacing']
        [nx, ny, nz] = self.params['NumberOfImage']
        dy = -1 * dy
        dz = -1 * dz
        Source_Init = np.array(self.params['SourceInit'])
        Detector_Init = np.array(self.params['DetectorInit'])
        StartAngle = self.params['StartAngle']
        EndAngle = self.params['EndAngle']
        Origin = np.array(self.params['Origin'])
        PhantomCenter = np.array(self.params['PhantomCenter'])
        HelicalPitch = self.params['Pitch']
        gpu = self.params['GPU']
        SAD = np.sqrt(np.sum((Source_Init - Origin) ** 2.0))
        SDD = np.sqrt(np.sum((Source_Init - Detector_Init) ** 2.0))
        if (HelicalPitch > 0):
            # P = HelicalPitch * (nv * dv * SAD) / SDD
            P = HelicalPitch * (nv * dv)
            nViews = (EndAngle - StartAngle) / (2 * pi) * nViews
            log.debug(nViews)
            assert (nViews % 1.0 == 0)
            nViews = int(nViews)
        else:
            P = 0.0
            # Calculates detector center
        angle = np.linspace(StartAngle, EndAngle, nViews + 1)
        angle = angle[0:-1]
        proj = np.zeros([nViews, nv, nu], dtype=np.float32)

        # Xplane = (PhantomCenter[0] - (nx - 1) / 2.0 + range(0, nx + 1)) * dx
        # Yplane = (PhantomCenter[1] - (ny - 1) / 2.0 + range(0, ny + 1)) * dy
        # Zplane = (PhantomCenter[2] - (nz - 1) / 2.0 + range(0, nz + 1)) * dz
        Xplane = PhantomCenter[0] + (np.arange(0, nx + 1) - (nx - 1) / 2.0) * dx
        Yplane = PhantomCenter[1] + (np.arange(0, ny + 1) - (ny - 1) / 2.0) * dy
        Zplane = PhantomCenter[2] + (np.arange(0, nz + 1) - (nz - 1) / 2.0) * dz
        Xplane = Xplane - dx / 2
        Yplane = Yplane - dy / 2
        Zplane = Zplane - dz / 2
        # print(Yplane[1]-Yplane[0])
        # print(Zplane[1]-Zplane[0])
        alpha = 0
        beta = 0
        gamma = 0
        eu = [cos(gamma) * cos(alpha), sin(alpha), sin(gamma)]
        ev = [cos(gamma) * -sin(alpha), cos(gamma) * cos(alpha), sin(gamma)]
        ew = [0, 0, 1]
        # print('Variable initialization: ' + str(time.time() - start_time))

        for i in range(nViews):
            # for i in range(12, 13):
            print(i)
            start_time = time.time()
            Source = np.array([-SAD * sin(angle[i]), SAD * cos(angle[i]),
                               Source_Init[2] + P * angle[i] / (2 * pi)])  # z-direction rotation
            Detector = np.array(
                [(SDD - SAD) * sin(angle[i]), -(SDD - SAD) * cos(angle[i]), Detector_Init[2] + P * angle[i] / (2 * pi)])
            # DetectorLength = np.array(
            #    [np.arange(floor(-nu / 2), floor(nu / 2) + 1) * du, np.arange(floor(-nv / 2), floor(nv / 2) + 1) * dv])
            # DetectorVectors = [eu, ev, ew]
            if (self.params['DetectorShape'] == 'Flat'):
                [DetectorIndex, DetectorBoundary] = self.FlatDetectorConstruction(Source, Detector, SDD, angle[i])
                # print(DetectorBoundary.shape)
                # print(DetectorIndex[0,128,:])
                # sys.exit()
            elif (self.params['DetectorShape'] == 'Curved'):
                [DetectorIndex, DetectorBoundary] = self.CurvedDetectorConstruction(Source, Detector, SDD, angle[i])
            else:
                print('Detector shape is not supproted!')
                sys.exit()
            # print('Detector initialization: ' + str(time.time() - start_time))
            if (self.params['Method'] == 'Distance'):
                start_time = time.time()
                proj[i, :, :] = self.distance(DetectorIndex, DetectorBoundary, Source, Detector, angle[i], Xplane,
                                              Yplane, Zplane)
                # print('Total projection: ' + str(time.time() - start_time))
            elif (self.params['Method'] == 'Ray'):
                proj[i, :, :] = self.ray(DetectorIndex, Source, Detector, angle[i], Xplane, Yplane, Zplane)
            # print('time taken: ' + str(time.time() - start_time) + '\n')
        self.proj = proj

    def forward(self):
        if (self.Method == 'Distance'):

            proj = self.distance_forward()

        elif (self.Method == 'Ray'):
            pass
            # proj[i, :, :] = self.ray(DetectorIndex, Source, Detector, angle[i], Xplane, Yplane, Zplane)

        self.proj = proj

    def distance_forward(self):
        nu = self.nu
        nv = self.nv
        du = self.du
        dv = -1 * self.dv
        nx = self.nx
        ny = self.ny
        nz = self.nz
        dx = self.dx
        dy = -1 * self.dy
        dz = -1 * self.dz
        nViews = self.nView

        sAngle = self.sAngle
        eAngle = self.eAngle
        angle = np.linspace(sAngle, eAngle, nViews + 1)
        Source_Init = self.Source
        Detector_Init = self.Detector
        Origin = self.Origin
        source_z0 = self.source_z0
        detector_z0 = self.detector_z0
        H = self.HelicalTrans
        PhantomCenter = self.PhantomCenter

        SAD = self.SAD
        SDD = self.SDD
        angle = np.linspace(sAngle, eAngle, nViews + 1)
        angle = angle[0:-1]
        proj = np.zeros([nViews, nv, nu], dtype=np.float32)

        Xplane = PhantomCenter[0] + (np.arange(0, nx + 1) - (nx - 1) / 2.0) * dx
        Yplane = PhantomCenter[1] + (np.arange(0, ny + 1) - (ny - 1) / 2.0) * dy
        Zplane = PhantomCenter[2] + (np.arange(0, nz + 1) - (nz - 1) / 2.0) * dz
        Xplane = Xplane - dx / 2
        Yplane = Yplane - dy / 2
        Zplane = Zplane - dz / 2
        if self.GPU:
            device = drv.Device(0)
            attrs = device.get_attributes()
            MAX_THREAD_PER_BLOCK = attrs[pycuda._driver.device_attribute.MAX_THREADS_PER_BLOCK]
            MAX_GRID_DIM_X = attrs[pycuda._driver.device_attribute.MAX_GRID_DIM_X]
            distance_proj_on_y_gpu = mod.get_function("distance_project_on_y3")
            distance_proj_on_x_gpu = mod.get_function("distance_project_on_x3")
            distance_proj_on_z_gpu = mod.get_function("distance_project_on_z3")
            image = np.copy(self.image)
            image_gpu = pycuda.gpuarray.to_gpu(image.flatten().astype(np.float32))
            dest = pycuda.gpuarray.to_gpu(proj.flatten().astype(np.float32))
            x_plane_gpu = pycuda.gpuarray.to_gpu(Xplane.astype(np.float32))
            y_plane_gpu = pycuda.gpuarray.to_gpu(Yplane.astype(np.float32))
            z_plane_gpu = pycuda.gpuarray.to_gpu(Zplane.astype(np.float32))
            slope_x1_gpu = pycuda.gpuarray.zeros(nu * nv, np.float32)
            slope_x2_gpu = pycuda.gpuarray.zeros(nu * nv, np.float32)
            slope_y1_gpu = pycuda.gpuarray.zeros(nu * nv, np.float32)
            slope_y2_gpu = pycuda.gpuarray.zeros(nu * nv, np.float32)
            slope_z1_gpu = pycuda.gpuarray.zeros(nu * nv, np.float32)
            slope_z2_gpu = pycuda.gpuarray.zeros(nu * nv, np.float32)
            intercept_x1_gpu = pycuda.gpuarray.zeros(nu * nv, np.float32)
            intercept_x2_gpu = pycuda.gpuarray.zeros(nu * nv, np.float32)
            intercept_y1_gpu = pycuda.gpuarray.zeros(nu * nv, np.float32)
            intercept_y2_gpu = pycuda.gpuarray.zeros(nu * nv, np.float32)
            intercept_z1_gpu = pycuda.gpuarray.zeros(nu * nv, np.float32)
            intercept_z2_gpu = pycuda.gpuarray.zeros(nu * nv, np.float32)
            intersection_gpu = pycuda.gpuarray.zeros(nu * nv, np.float32)
            proj_param_gpu = pycuda.gpuarray.zeros(9, dtype=np.float32)
        for i in range(nViews):
            log.debug(i)
            Source = np.array([-SAD * sin(angle[i]), SAD * cos(angle[i]), source_z0 + H * angle[i] / (2 * pi)])
            Detector = np.array(
                [(SDD - SAD) * sin(angle[i]), -(SDD - SAD) * cos(angle[i]), detector_z0 + H * angle[i] / (2 * pi)])
            if self.DetectorShape == 'Flat':
                [DetectorIndex, DetectorBoundary] = self.FlatDetectorConstruction(Source, Detector, SDD, angle[i])

            elif self.DetectorShape == 'Curved':
                [DetectorIndex, DetectorBoundary] = self.CurvedDetectorConstruction(Source, Detector, SDD, angle[i])
            else:
                print('Detector shape is not supproted!')
                sys.exit()

            DetectorBoundaryU1 = np.array(
                [DetectorBoundary[0, 0:-1, 0:-1], DetectorBoundary[1, 0:-1, 0:-1], DetectorIndex[2, :, :]])
            DetectorBoundaryU2 = np.array(
                [DetectorBoundary[0, 1:, 1:], DetectorBoundary[1, 1:, 1:], DetectorIndex[2, :, :]])
            DetectorBoundaryV1 = np.array([DetectorIndex[0, :, :], DetectorIndex[1, :, :], DetectorBoundary[2, 1:, 1:]])
            DetectorBoundaryV2 = np.array(
                [DetectorIndex[0, :, :], DetectorIndex[1, :, :], DetectorBoundary[2, 0:-1, 0:-1]])
            ray_angles = atan(sqrt(
                (DetectorIndex[0, :, :] - Detector[0]) ** 2.0 + (DetectorIndex[1, :, :] - Detector[1]) ** 2.0 + (
                        DetectorIndex[2, :, :] - Detector[2]) ** 2.0) / SDD)
            # ray_normalization = cos(ray_angles)
            ray_normalization = 1.0
            if abs(Source[0] - Detector[0]) >= abs(Source[1] - Detector[1]) and abs(Source[0] - Detector[0]) >= abs(
                    Source[2] - Detector[2]):
                SlopesU1 = (Source[1] - DetectorBoundaryU1[1, :, :]) / (Source[0] - DetectorBoundaryU1[0, :, :])
                InterceptsU1 = -SlopesU1 * Source[0] + Source[1]
                SlopesU2 = (Source[1] - DetectorBoundaryU2[1, :, :]) / (Source[0] - DetectorBoundaryU2[0, :, :])
                InterceptsU2 = -SlopesU2 * Source[0] + Source[1]
                SlopesV1 = (Source[2] - DetectorBoundaryV1[2, :, :]) / (Source[0] - DetectorBoundaryV1[0, :, :])
                InterceptsV1 = -SlopesV1 * Source[0] + Source[2]
                SlopesV2 = (Source[2] - DetectorBoundaryV2[2, :, :]) / (Source[0] - DetectorBoundaryV2[0, :, :])
                InterceptsV2 = -SlopesV2 * Source[0] + Source[2]
                intersection_slope1 = (Source[1] - DetectorIndex[1, :, :]) / (Source[0] - DetectorIndex[0, :, :])
                intersection_slope2 = (Source[2] - DetectorIndex[2, :, :]) / (Source[0] - DetectorIndex[0, :, :])
                intersection_length = abs(dx) / (cos(atan(intersection_slope1)) * cos(atan(intersection_slope2)))

                if (self.GPU):
                    TotalSize = nu * nv * nx
                    if (TotalSize < MAX_THREAD_PER_BLOCK):
                        blockX = nu * nv * nx
                        blockY = 1
                        blockZ = 1
                        gridX = 1
                        gridY = 1
                    else:
                        blockX = 32
                        blockY = 32
                        blockZ = 1
                        GridSize = ceil(TotalSize / (blockX * blockY))
                        try:
                            if (GridSize < MAX_GRID_DIM_X):
                                [gridX, gridY] = Reconstruction._optimalGrid(GridSize)
                            else:
                                raise ErrorDescription(6)
                        except ErrorDescription as e:
                            print(e)
                            sys.exit()

                    proj_param = np.array([dx, dy, dz, nx, ny, nz, nu, nv, i]).astype(np.float32)
                    slope_y1_gpu.set(SlopesU1.flatten().astype(np.float32))
                    slope_y2_gpu.set(SlopesU2.flatten().astype(np.float32))
                    slope_z1_gpu.set(SlopesV1.flatten().astype(np.float32))
                    slope_z2_gpu.set(SlopesV2.flatten().astype(np.float32))
                    intercept_y1_gpu.set(InterceptsU1.flatten().astype(np.float32))
                    intercept_y2_gpu.set(InterceptsU2.flatten().astype(np.float32))
                    intercept_z1_gpu.set(InterceptsV1.flatten().astype(np.float32))
                    intercept_z2_gpu.set(InterceptsV2.flatten().astype(np.float32))
                    intersection_gpu.set(intersection_length.flatten().astype(np.float32))
                    proj_param_gpu.set(proj_param.flatten().astype(np.float32))

                    distance_proj_on_x_gpu(dest, image_gpu, slope_y1_gpu, slope_y2_gpu, slope_z1_gpu, slope_z2_gpu,
                                           intercept_y1_gpu, intercept_y2_gpu, intercept_z1_gpu, intercept_z2_gpu,
                                           x_plane_gpu, y_plane_gpu, z_plane_gpu, intersection_gpu, proj_param_gpu,
                                           block=(blockX, blockY, blockZ), grid=(gridX, gridY))

                else:
                    for ix in range(nx):
                        CoordY1 = SlopesU1 * (Xplane[ix] + dx / 2) + InterceptsU1
                        CoordY2 = SlopesU2 * (Xplane[ix] + dx / 2) + InterceptsU2
                        CoordZ1 = SlopesV1 * (Xplane[ix] + dx / 2) + InterceptsV1
                        CoordZ2 = SlopesV2 * (Xplane[ix] + dx / 2) + InterceptsV2
                        image_y1 = floor((CoordY1 - Yplane[0] + 0) / dy)
                        image_y2 = floor((CoordY2 - Yplane[0] + 0) / dy)
                        image_z1 = floor((CoordZ1 - Zplane[0] + 0) / dz)
                        image_z2 = floor((CoordZ2 - Zplane[0] + 0) / dz)
                        proj += self._distance_project_on_x(self.image, CoordY1, CoordY2, CoordZ1, CoordZ2, Yplane,
                                                            Zplane,
                                                            image_y1, image_y2, image_z1, image_z2, dy, dz, ix) * (
                                        intersection_length / ray_normalization)


            elif abs(Source[1] - Detector[1]) >= abs(Source[0] - Detector[0]) and abs(Source[1] - Detector[1]) >= abs(
                    Source[2] - Detector[2]):

                SlopesU1 = (Source[0] - DetectorBoundaryU1[0, :, :]) / (Source[1] - DetectorBoundaryU1[1, :, :])
                InterceptsU1 = -SlopesU1 * Source[1] + Source[0]
                SlopesU2 = (Source[0] - DetectorBoundaryU2[0, :, :]) / (Source[1] - DetectorBoundaryU2[1, :, :])
                InterceptsU2 = -SlopesU2 * Source[1] + Source[0]
                SlopesV1 = (Source[2] - DetectorBoundaryV1[2, :, :]) / (Source[1] - DetectorBoundaryV1[1, :, :])
                InterceptsV1 = -SlopesV1 * Source[1] + Source[2]
                SlopesV2 = (Source[2] - DetectorBoundaryV2[2, :, :]) / (Source[1] - DetectorBoundaryV2[1, :, :])
                InterceptsV2 = -SlopesV2 * Source[1] + Source[2]
                # print('Calculate line: ' + str(time.time() - start_time))
                intersection_slope1 = (Source[0] - DetectorIndex[0, :, :]) / (Source[1] - DetectorIndex[1, :, :])
                intersection_slope2 = (Source[2] - DetectorIndex[2, :, :]) / (Source[1] - DetectorIndex[1, :, :])
                intersection_length = abs(dy) / (cos(atan(intersection_slope1)) * cos(atan(intersection_slope2)))
                if (self.params['GPU']):
                    TotalSize = nu * nv * ny
                    if (TotalSize < MAX_THREAD_PER_BLOCK):
                        blockX = nu * nv * ny
                        blockY = 1
                        blockZ = 1
                        gridX = 1
                        gridY = 1
                    else:
                        blockX = 32
                        blockY = 32
                        blockZ = 1
                        GridSize = ceil(TotalSize / (blockX * blockY))
                        try:
                            if (GridSize < MAX_GRID_DIM_X):
                                [gridX, gridY] = Reconstruction._optimalGrid(GridSize)
                            else:
                                raise ErrorDescription(6)
                        except ErrorDescription as e:
                            print(e)
                            # slope_x1_gpu=pycuda.gpuarray.to_gpu(SlopesU1.flatten().astype(np.float32))
                            sys.exit()

                    proj_param = np.array([dx, dy, dz, nx, ny, nz, nu, nv, i]).astype(np.float32)

                    slope_x1_gpu.set(SlopesU1.flatten().astype(np.float32))
                    slope_x2_gpu.set(SlopesU2.flatten().astype(np.float32))
                    slope_z1_gpu.set(SlopesV1.flatten().astype(np.float32))
                    slope_z2_gpu.set(SlopesV2.flatten().astype(np.float32))
                    intercept_x1_gpu.set(InterceptsU1.flatten().astype(np.float32))
                    intercept_x2_gpu.set(InterceptsU2.flatten().astype(np.float32))
                    intercept_z1_gpu.set(InterceptsV1.flatten().astype(np.float32))
                    intercept_z2_gpu.set(InterceptsV2.flatten().astype(np.float32))
                    intersection_gpu.set(intersection_length.flatten().astype(np.float32))
                    proj_param_gpu.set(proj_param.flatten().astype(np.float32))
                    distance_proj_on_y_gpu(dest, image_gpu, slope_x1_gpu, slope_x2_gpu, slope_z1_gpu, slope_z2_gpu,
                                           intercept_x1_gpu, intercept_x2_gpu, intercept_z1_gpu, intercept_z2_gpu,
                                           x_plane_gpu, y_plane_gpu, z_plane_gpu, intersection_gpu, proj_param_gpu,
                                           block=(blockX, blockY, blockZ), grid=(gridX, gridY))

                else:
                    for iy in range(ny):
                        start_time = time.time()
                        CoordX1 = SlopesU1 * (Yplane[iy] + dy / 2) + InterceptsU1
                        CoordX2 = SlopesU2 * (Yplane[iy] + dy / 2) + InterceptsU2
                        CoordZ1 = SlopesV1 * (Yplane[iy] + dy / 2) + InterceptsV1
                        CoordZ2 = SlopesV2 * (Yplane[iy] + dy / 2) + InterceptsV2
                        image_x1 = floor((CoordX1 - Xplane[0] + 0) / dx)
                        image_x2 = floor((CoordX2 - Xplane[0] + 0) / dx)
                        image_z1 = floor((CoordZ1 - Zplane[0] + 0) / dz)
                        image_z2 = floor((CoordZ2 - Zplane[0] + 0) / dz)
                        proj[i, :, :] += self._distance_project_on_y(self.image, CoordX1, CoordX2, CoordZ1, CoordZ2,
                                                                     Xplane, Zplane, image_x1, image_x2, image_z1,
                                                                     image_z2, dx, dz, iy) * (
                                                 intersection_length / ray_normalization)

            else:
                SlopesU1 = (Source[0] - DetectorBoundaryU1[0, :, :]) / (Source[2] - DetectorBoundaryU1[2, :, :])
                InterceptsU1 = -SlopesU1 * Source[2] + Source[0]
                SlopesU2 = (Source[0] - DetectorBoundaryU2[0, :, :]) / (Source[2] - DetectorBoundaryU2[2, :, :])
                InterceptsU2 = -SlopesU2 * Source[2] + Source[0]
                SlopesV1 = (Source[1] - DetectorBoundaryV1[1, :, :]) / (Source[2] - DetectorBoundaryV1[2, :, :])
                InterceptsV1 = -SlopesV1 * Source[2] + Source[1]
                SlopesV2 = (Source[1] - DetectorBoundaryV2[1, :, :]) / (Source[2] - DetectorBoundaryV2[2, :, :])
                InterceptsV2 = -SlopesV2 * Source[2] + Source[1]
                intersection_slope1 = (Source[0] - DetectorIndex[0, :, :]) / (Source[2] - DetectorIndex[2, :, :])
                intersection_slope2 = (Source[1] - DetectorIndex[1, :, :]) / (Source[2] - DetectorIndex[2, :, :])
                intersection_length = abs(dz) / (cos(atan(intersection_slope1)) * cos(atan(intersection_slope2)))
                if (self.params['GPU']):
                    TotalSize = nu * nv * nz
                    if (TotalSize < MAX_THREAD_PER_BLOCK):
                        blockX = nu * nv * nz
                        blockY = 1
                        blockZ = 1
                        gridX = 1
                        gridY = 1
                    else:
                        blockX = 32
                        blockY = 32
                        blockZ = 1
                        GridSize = ceil(TotalSize / (blockX * blockY))
                        try:
                            if (GridSize < MAX_GRID_DIM_X):
                                [gridX, gridY] = Reconstruction._optimalGrid(GridSize)
                            else:
                                raise ErrorDescription(6)
                        except ErrorDescription as e:
                            print(e)
                            sys.exit()

                    proj_param = np.array([dx, dy, dz, nx, ny, nz, nu, nv, i]).astype(np.float32)

                    slope_x1_gpu.set(SlopesU1.flatten().astype(np.float32))
                    slope_x2_gpu.set(SlopesU2.flatten().astype(np.float32))
                    slope_y1_gpu.set(SlopesV1.flatten().astype(np.float32))
                    slope_y2_gpu.set(SlopesV2.flatten().astype(np.float32))
                    intercept_x1_gpu.set(InterceptsU1.flatten().astype(np.float32))
                    intercept_x2_gpu.set(InterceptsU2.flatten().astype(np.float32))
                    intercept_y1_gpu.set(InterceptsV1.flatten().astype(np.float32))
                    intercept_y2_gpu.set(InterceptsV2.flatten().astype(np.float32))
                    intersection_gpu.copy_to_device(intersection_length.flatten().astype(np.float32))
                    proj_param_gpu.set(proj_param.flatten().astype(np.float32))
                    distance_proj_on_z_gpu(dest, image_gpu, slope_x1_gpu, slope_x2_gpu, slope_y1_gpu, slope_y2_gpu,
                                           intercept_x1_gpu, intercept_x2_gpu, intercept_y1_gpu, intercept_y2_gpu,
                                           x_plane_gpu, y_plane_gpu, z_plane_gpu, intersection_gpu, proj_param_gpu,
                                           block=(blockX, blockY, blockZ), grid=(gridX, gridY))

                else:
                    for iz in range(nz):
                        CoordX1 = SlopesU1 * Zplane[iz] + dz / 2 + InterceptsU1
                        CoordX2 = SlopesU2 * Zplane[iz] + dz / 2 + InterceptsU2
                        CoordY1 = SlopesV1 * Zplane[iz] + dz / 2 + InterceptsV1
                        CoordY2 = SlopesV2 * Zplane[iz] + dz / 2 + InterceptsV2
                        image_x1 = floor(CoordX1 - Xplane[0] + dx) / dx
                        image_x2 = floor(CoordX2 - Xplane[0] + dx) / dx
                        image_y1 = floor(CoordY1 - Yplane[0] + dy) / dy
                        image_y2 = floor(CoordY2 - Yplane[0] + dy) / dy
                        proj[i, :, :] += self._distance_project_on_z(self.image, CoordX1, CoordX2, CoordY1, CoordY2,
                                                                     Xplane, Yplane, image_x1, image_x2, image_y1,
                                                                     image_y2, dx, dy, iz) * (
                                                 intersection_length / ray_normalization)
        if self.GPU:
            proj = dest.get().reshape([nViews, nv, nu]).astype(np.float32)
        return proj

    def distance(self, DetectorIndex, DetectorBoundary, Source, Detector, angle, Xplane, Yplane, Zplane):
        [nu, nv] = self.params['NumberOfDetectorPixels']
        [du, dv] = self.params['DetectorPixelSize']
        [dx, dy, dz] = self.params['ImagePixelSpacing']
        [nx, ny, nz] = self.params['NumberOfImage']
        dy = -1 * dy
        dz = -1 * dz
        dv = -1 * dv
        proj = np.zeros([nv, nu], dtype=np.float32)
        if self.params['GPU']:
            device = drv.Device(0)
            attrs = device.get_attributes()
            MAX_THREAD_PER_BLOCK = attrs[pycuda._driver.device_attribute.MAX_THREADS_PER_BLOCK]
            MAX_GRID_DIM_X = attrs[pycuda._driver.device_attribute.MAX_GRID_DIM_X]
            distance_proj_on_y_gpu = mod.get_function("distance_project_on_y2")
            distance_proj_on_x_gpu = mod.get_function("distance_project_on_x2")
            distance_proj_on_z_gpu = mod.get_function("distance_project_on_z2")
            image = self.image.flatten().astype(np.float32)
            dest = pycuda.gpuarray.to_gpu(proj.flatten().astype(np.float32))
            x_plane_gpu = pycuda.gpuarray.to_gpu(Xplane.astype(np.float32))
            y_plane_gpu = pycuda.gpuarray.to_gpu(Yplane.astype(np.float32))
            z_plane_gpu = pycuda.gpuarray.to_gpu(Zplane.astype(np.float32))
        start_time = time.time()
        DetectorBoundaryU1 = np.array(
            [DetectorBoundary[0, 0:-1, 0:-1], DetectorBoundary[1, 0:-1, 0:-1], DetectorIndex[2, :, :]])
        DetectorBoundaryU2 = np.array(
            [DetectorBoundary[0, 1:, 1:], DetectorBoundary[1, 1:, 1:], DetectorIndex[2, :, :]])
        DetectorBoundaryV1 = np.array([DetectorIndex[0, :, :], DetectorIndex[1, :, :], DetectorBoundary[2, 1:, 1:]])
        DetectorBoundaryV2 = np.array([DetectorIndex[0, :, :], DetectorIndex[1, :, :], DetectorBoundary[2, 0:-1, 0:-1]])
        # DetectorBoundaryU1 = np.array(
        #     [DetectorIndex[0, :, :] - cos(angle) * du / 2, DetectorIndex[1, :, :] - sin(angle) * du / 2,
        #      DetectorIndex[2, :, :]])
        # DetectorBoundaryU2 = np.array(
        #     [DetectorIndex[0, :, :] + cos(angle) * du / 2, DetectorIndex[1, :, :] + sin(angle) * du / 2,
        #      DetectorIndex[2, :, :]])
        # DetectorBoundaryV1 = np.array([DetectorIndex[0, :, :], DetectorIndex[1, :, :], DetectorIndex[2, :, :] - dv / 2])
        # DetectorBoundaryV2 = np.array([DetectorIndex[0, :, :], DetectorIndex[1, :, :], DetectorIndex[2, :, :] + dv / 2])
        SDD = sqrt(np.sum((Source - Detector) ** 2.0))
        ray_angles = atan(sqrt(
            (DetectorIndex[0, :, :] - Detector[0]) ** 2.0 + (DetectorIndex[1, :, :] - Detector[1]) ** 2.0 + (
                    DetectorIndex[2, :, :] - Detector[2]) ** 2.0) / SDD)
        # ray_normalization = cos(ray_angles)
        ray_normalization = 1.0
        if (abs(Source[0] - Detector[0]) >= abs(Source[1] - Detector[1]) and abs(Source[0] - Detector[0]) >= abs(
                Source[2] - Detector[2])):
            SlopesU1 = (Source[1] - DetectorBoundaryU1[1, :, :]) / (Source[0] - DetectorBoundaryU1[0, :, :])
            InterceptsU1 = -SlopesU1 * Source[0] + Source[1]
            SlopesU2 = (Source[1] - DetectorBoundaryU2[1, :, :]) / (Source[0] - DetectorBoundaryU2[0, :, :])
            InterceptsU2 = -SlopesU2 * Source[0] + Source[1]
            SlopesV1 = (Source[2] - DetectorBoundaryV1[2, :, :]) / (Source[0] - DetectorBoundaryV1[0, :, :])
            InterceptsV1 = -SlopesV1 * Source[0] + Source[2]
            SlopesV2 = (Source[2] - DetectorBoundaryV2[2, :, :]) / (Source[0] - DetectorBoundaryV2[0, :, :])
            InterceptsV2 = -SlopesV2 * Source[0] + Source[2]
            intersection_slope1 = (Source[1] - DetectorIndex[1, :, :]) / (Source[0] - DetectorIndex[0, :, :])
            intersection_slope2 = (Source[2] - DetectorIndex[2, :, :]) / (Source[0] - DetectorIndex[0, :, :])
            intersection_length = abs(dx) / (cos(atan(intersection_slope1)) * cos(atan(intersection_slope2)))

            if (self.params['GPU']):
                TotalSize = nu * nv * nx
                if (TotalSize < MAX_THREAD_PER_BLOCK):
                    blockX = nu * nv * nx
                    blockY = 1
                    blockZ = 1
                    gridX = 1
                    gridY = 1
                else:
                    blockX = 32
                    blockY = 32
                    blockZ = 1
                    GridSize = ceil(TotalSize / (blockX * blockY))
                    try:
                        if (GridSize < MAX_GRID_DIM_X):
                            [gridX, gridY] = Reconstruction._optimalGrid(GridSize)
                        else:
                            raise ErrorDescription(6)
                    except ErrorDescription as e:
                        print(e)
                        sys.exit()
                proj_param = np.array([dx, dy, dz, nx, ny, nz, nu, nv]).astype(np.float32)
                slope_y1_gpu = pycuda.gpuarray.to_gpu(SlopesU1.flatten().astype(np.float32))
                slope_y2_gpu = pycuda.gpuarray.to_gpu(SlopesU2.flatten().astype(np.float32))
                slope_z1_gpu = pycuda.gpuarray.to_gpu(SlopesV1.flatten().astype(np.float32))
                slope_z2_gpu = pycuda.gpuarray.to_gpu(SlopesV2.flatten().astype(np.float32))
                intercept_y1_gpu = pycuda.gpuarray.to_gpu(InterceptsU1.flatten().astype(np.float32))
                intercept_y2_gpu = pycuda.gpuarray.to_gpu(InterceptsU2.flatten().astype(np.float32))
                intercept_z1_gpu = pycuda.gpuarray.to_gpu(InterceptsV1.flatten().astype(np.float32))
                intercept_z2_gpu = pycuda.gpuarray.to_gpu(InterceptsV2.flatten().astype(np.float32))
                proj_param_gpu = pycuda.gpuarray.to_gpu(proj_param)
                distance_proj_on_x_gpu(dest, drv.In(image), slope_y1_gpu, slope_y2_gpu, slope_z1_gpu,
                                       slope_z2_gpu, intercept_y1_gpu, intercept_y2_gpu, intercept_z1_gpu,
                                       intercept_z2_gpu, x_plane_gpu, y_plane_gpu, z_plane_gpu, proj_param_gpu,
                                       block=(blockX, blockY, blockZ), grid=(gridX, gridY))
                del slope_y1_gpu, slope_y2_gpu, slope_z1_gpu, slope_z2_gpu, intercept_y1_gpu, intercept_y2_gpu, intercept_z1_gpu, intercept_z2_gpu, x_plane_gpu, y_plane_gpu, z_plane_gpu
                proj = dest.get().reshape([nv, nu]).astype(np.float32)
                proj = proj * (intersection_length / ray_normalization)
                del dest
            else:
                for ix in range(nx):
                    CoordY1 = SlopesU1 * (Xplane[ix] + dx / 2) + InterceptsU1
                    CoordY2 = SlopesU2 * (Xplane[ix] + dx / 2) + InterceptsU2
                    CoordZ1 = SlopesV1 * (Xplane[ix] + dx / 2) + InterceptsV1
                    CoordZ2 = SlopesV2 * (Xplane[ix] + dx / 2) + InterceptsV2
                    image_y1 = floor((CoordY1 - Yplane[0] + 0) / dy)
                    image_y2 = floor((CoordY2 - Yplane[0] + 0) / dy)
                    image_z1 = floor((CoordZ1 - Zplane[0] + 0) / dz)
                    image_z2 = floor((CoordZ2 - Zplane[0] + 0) / dz)
                    proj += self._distance_project_on_x(self.image, CoordY1, CoordY2, CoordZ1, CoordZ2, Yplane, Zplane,
                                                        image_y1, image_y2, image_z1, image_z2, dy, dz, ix) * (
                                    intersection_length / ray_normalization)


        elif (abs(Source[1] - Detector[1]) >= abs(Source[0] - Detector[0]) and abs(Source[1] - Detector[1]) >= abs(
                Source[2] - Detector[2])):
            start_time = time.time()
            SlopesU1 = (Source[0] - DetectorBoundaryU1[0, :, :]) / (Source[1] - DetectorBoundaryU1[1, :, :])
            InterceptsU1 = -SlopesU1 * Source[1] + Source[0]
            SlopesU2 = (Source[0] - DetectorBoundaryU2[0, :, :]) / (Source[1] - DetectorBoundaryU2[1, :, :])
            InterceptsU2 = -SlopesU2 * Source[1] + Source[0]
            SlopesV1 = (Source[2] - DetectorBoundaryV1[2, :, :]) / (Source[1] - DetectorBoundaryV1[1, :, :])
            InterceptsV1 = -SlopesV1 * Source[1] + Source[2]
            SlopesV2 = (Source[2] - DetectorBoundaryV2[2, :, :]) / (Source[1] - DetectorBoundaryV2[1, :, :])
            InterceptsV2 = -SlopesV2 * Source[1] + Source[2]
            # print('Calculate line: ' + str(time.time() - start_time))
            intersection_slope1 = (Source[0] - DetectorIndex[0, :, :]) / (Source[1] - DetectorIndex[1, :, :])
            intersection_slope2 = (Source[2] - DetectorIndex[2, :, :]) / (Source[1] - DetectorIndex[1, :, :])
            intersection_length = abs(dy) / (cos(atan(intersection_slope1)) * cos(atan(intersection_slope2)))
            if (self.params['GPU']):
                TotalSize = nu * nv * ny
                if (TotalSize < MAX_THREAD_PER_BLOCK):
                    blockX = nu * nv * ny
                    blockY = 1
                    blockZ = 1
                    gridX = 1
                    gridY = 1
                else:
                    blockX = 32
                    blockY = 32
                    blockZ = 1
                    GridSize = ceil(TotalSize / (blockX * blockY))
                    try:
                        if (GridSize < MAX_GRID_DIM_X):
                            [gridX, gridY] = Reconstruction._optimalGrid(GridSize)
                        else:
                            raise ErrorDescription(6)
                    except ErrorDescription as e:
                        print(e)
                        sys.exit()
                proj_param = np.array([dx, dy, dz, nx, ny, nz, nu, nv]).astype(np.float32)
                slope_x1_gpu = pycuda.gpuarray.to_gpu(SlopesU1.flatten().astype(np.float32))
                slope_x2_gpu = pycuda.gpuarray.to_gpu(SlopesU2.flatten().astype(np.float32))
                slope_z1_gpu = pycuda.gpuarray.to_gpu(SlopesV1.flatten().astype(np.float32))
                slope_z2_gpu = pycuda.gpuarray.to_gpu(SlopesV2.flatten().astype(np.float32))
                intercept_x1_gpu = pycuda.gpuarray.to_gpu(InterceptsU1.flatten().astype(np.float32))
                intercept_x2_gpu = pycuda.gpuarray.to_gpu(InterceptsU2.flatten().astype(np.float32))
                intercept_z1_gpu = pycuda.gpuarray.to_gpu(InterceptsV1.flatten().astype(np.float32))
                intercept_z2_gpu = pycuda.gpuarray.to_gpu(InterceptsV2.flatten().astype(np.float32))
                proj_param_gpu = pycuda.gpuarray.to_gpu(proj_param)
                distance_proj_on_y_gpu(dest, drv.In(image), slope_x1_gpu, slope_x2_gpu, slope_z1_gpu,
                                       slope_z2_gpu, intercept_x1_gpu, intercept_x2_gpu, intercept_z1_gpu,
                                       intercept_z2_gpu, x_plane_gpu, y_plane_gpu, z_plane_gpu, proj_param_gpu,
                                       block=(blockX, blockY, blockZ), grid=(gridX, gridY))
                del slope_x1_gpu, slope_x2_gpu, slope_z1_gpu, slope_z2_gpu, intercept_x1_gpu, intercept_x2_gpu, intercept_z1_gpu, intercept_z2_gpu, x_plane_gpu, y_plane_gpu, z_plane_gpu
                proj = dest.get().reshape([nv, nu]).astype(np.float32)
                proj = proj * (intersection_length / ray_normalization)
                del dest
            else:
                for iy in range(ny):
                    start_time = time.time()
                    CoordX1 = SlopesU1 * (Yplane[iy] + dy / 2) + InterceptsU1
                    CoordX2 = SlopesU2 * (Yplane[iy] + dy / 2) + InterceptsU2
                    CoordZ1 = SlopesV1 * (Yplane[iy] + dy / 2) + InterceptsV1
                    CoordZ2 = SlopesV2 * (Yplane[iy] + dy / 2) + InterceptsV2
                    image_x1 = floor((CoordX1 - Xplane[0] + 0) / dx)
                    image_x2 = floor((CoordX2 - Xplane[0] + 0) / dx)
                    image_z1 = floor((CoordZ1 - Zplane[0] + 0) / dz)
                    image_z2 = floor((CoordZ2 - Zplane[0] + 0) / dz)
                    proj += self._distance_project_on_y(self.image, CoordX1, CoordX2, CoordZ1, CoordZ2, Xplane, Zplane,
                                                        image_x1, image_x2, image_z1, image_z2, dx, dz, iy) * (
                                    intersection_length / ray_normalization)

        else:
            SlopesU1 = (Source[0] - DetectorBoundaryU1[0, :, :]) / (Source[2] - DetectorBoundaryU1[2, :, :])
            InterceptsU1 = -SlopesU1 * Source[2] + Source[0]
            SlopesU2 = (Source[0] - DetectorBoundaryU2[0, :, :]) / (Source[2] - DetectorBoundaryU2[2, :, :])
            InterceptsU2 = -SlopesU2 * Source[2] + Source[0]
            SlopesV1 = (Source[1] - DetectorBoundaryV1[1, :, :]) / (Source[2] - DetectorBoundaryV1[2, :, :])
            InterceptsV1 = -SlopesV1 * Source[2] + Source[1]
            SlopesV2 = (Source[1] - DetectorBoundaryV2[1, :, :]) / (Source[2] - DetectorBoundaryV2[2, :, :])
            InterceptsV2 = -SlopesV2 * Source[2] + Source[1]
            intersection_slope1 = (Source[0] - DetectorIndex[0, :, :]) / (Source[2] - DetectorIndex[2, :, :])
            intersection_slope2 = (Source[1] - DetectorIndex[1, :, :]) / (Source[2] - DetectorIndex[2, :, :])
            intersection_length = abs(dz) / (cos(atan(intersection_slope1)) * cos(atan(intersection_slope2)))
            if (self.params['GPU']):
                TotalSize = nu * nv * nz
                if (TotalSize < MAX_THREAD_PER_BLOCK):
                    blockX = nu * nv * nz
                    blockY = 1
                    blockZ = 1
                    gridX = 1
                    gridY = 1
                else:
                    blockX = 32
                    blockY = 32
                    blockZ = 1
                    GridSize = ceil(TotalSize / (blockX * blockY))
                    try:
                        if (GridSize < MAX_GRID_DIM_X):
                            [gridX, gridY] = Reconstruction._optimalGrid(GridSize)
                        else:
                            raise ErrorDescription(6)
                    except ErrorDescription as e:
                        print(e)
                        sys.exit()
                proj_param = np.array([dx, dy, dz, nx, ny, nz, nu, nv]).astype(np.float32)
                slope_x1_gpu = pycuda.gpuarray.to_gpu(SlopesU1.flatten().astype(np.float32))
                slope_x2_gpu = pycuda.gpuarray.to_gpu(SlopesU2.flatten().astype(np.float32))
                slope_y1_gpu = pycuda.gpuarray.to_gpu(SlopesV1.flatten().astype(np.float32))
                slope_y2_gpu = pycuda.gpuarray.to_gpu(SlopesV2.flatten().astype(np.float32))
                intercept_x1_gpu = pycuda.gpuarray.to_gpu(InterceptsU1.flatten().astype(np.float32))
                intercept_x2_gpu = pycuda.gpuarray.to_gpu(InterceptsU2.flatten().astype(np.float32))
                intercept_y1_gpu = pycuda.gpuarray.to_gpu(InterceptsV1.flatten().astype(np.float32))
                intercept_y2_gpu = pycuda.gpuarray.to_gpu(InterceptsV2.flatten().astype(np.float32))
                proj_param_gpu = pycuda.gpuarray.to_gpu(proj_param)
                distance_proj_on_z_gpu(dest, drv.In(image), slope_x1_gpu, slope_x2_gpu, slope_y1_gpu,
                                       slope_y2_gpu, intercept_x1_gpu, intercept_x2_gpu, intercept_y1_gpu,
                                       intercept_y2_gpu, x_plane_gpu, y_plane_gpu, z_plane_gpu, proj_param_gpu,
                                       block=(blockX, blockY, blockZ), grid=(gridX, gridY))
                del slope_x1_gpu, slope_x2_gpu, slope_y1_gpu, slope_y2_gpu, intercept_x1_gpu, intercept_x2_gpu, intercept_y1_gpu, intercept_y2_gpu, x_plane_gpu, y_plane_gpu, z_plane_gpu
                proj = dest.get().reshape([nv, nu]).astype(np.float32)
                proj = proj * (intersection_length / ray_normalization)
                del dest
            else:
                for iz in range(nz):
                    CoordX1 = SlopesU1 * Zplane[iz] + dz / 2 + InterceptsU1
                    CoordX2 = SlopesU2 * Zplane[iz] + dz / 2 + InterceptsU2
                    CoordY1 = SlopesV1 * Zplane[iz] + dz / 2 + InterceptsV1
                    CoordY2 = SlopesV2 * Zplane[iz] + dz / 2 + InterceptsV2
                    image_x1 = floor(CoordX1 - Xplane[0] + dx) / dx
                    image_x2 = floor(CoordX2 - Xplane[0] + dx) / dx
                    image_y1 = floor(CoordY1 - Yplane[0] + dy) / dy
                    image_y2 = floor(CoordY2 - Yplane[0] + dy) / dy
                    proj += self._distance_project_on_z(self.image, CoordX1, CoordX2, CoordY1, CoordY2, Xplane, Yplane,
                                                        image_x1, image_x2, image_y1, image_y2, dx, dy, iz) * (
                                    intersection_length / ray_normalization)
        return proj

    @staticmethod
    def _distance_project_on_y(image, CoordX1, CoordX2, CoordZ1, CoordZ2, Xplane, Zplane, image_x1, image_x2, image_z1,
                               image_z2, dx, dz, iy):
        tol_min = 1e-6
        tol_max = 1e7
        proj = np.zeros(CoordX1.shape, dtype=np.float32)
        start_time = time.time()
        for i in range(CoordX1.shape[0]):
            for j in range(CoordX1.shape[1]):
                p_value = 0
                s_index_x = min(image_x1[i, j], image_x2[i, j])
                e_index_x = max(image_x1[i, j], image_x2[i, j])
                s_index_z = min(image_z1[i, j], image_z2[i, j])
                e_index_z = max(image_z2[i, j], image_z2[i, j])
                for k in range(int(s_index_x), int(e_index_x) + 1):
                    if (k < 0 or k > image.shape[0] - 1):
                        continue
                    if (s_index_x == e_index_x):
                        weight1 = 1
                    elif (k == s_index_x):
                        # print(k,s_index_x,e_index_x,Xplane[k+1],CoordX1[i,j],CoordX2[i,j])
                        weight1 = (Xplane[k + 1] - min(CoordX1[i, j], CoordX2[i, j])) / abs(
                            CoordX1[i, j] - CoordX2[i, j])
                    elif (k == e_index_x):
                        # print(k,s_index_x,e_index_x)
                        # print(Xplane[k],CoordX1[i,j],CoordX2[i,j])
                        weight1 = (max(CoordX1[i, j], CoordX2[i, j]) - Xplane[k]) / abs(CoordX1[i, j] - CoordX2[i, j])
                    else:
                        weight1 = abs(dx) / abs(CoordX1[i, j] - CoordX2[i, j])
                    for l in range(int(s_index_z), int(e_index_z) + 1):
                        if (l < 0 or l > image.shape[2] - 1):
                            continue
                        if (s_index_z == e_index_z):
                            weight2 = 1
                        elif (l == s_index_z):
                            # print(s_index_z,e_index_z,Zplane[l+1],CoordZ1[i,j],CoordZ2[i,j])
                            weight2 = (max(CoordZ1[i, j], CoordZ2[i, j]) - Zplane[l + 1]) / abs(
                                CoordZ1[i, j] - CoordZ2[i, j])
                        elif (l == e_index_z):
                            # print('1')
                            weight2 = (Zplane[l] - min(CoordZ1[i, j], CoordZ2[i, j])) / abs(
                                CoordZ1[i, j] - CoordZ2[i, j])
                        else:
                            weight2 = abs(dz) / abs(CoordZ1[i, j] - CoordZ2[i, j])
                        # print(weight1,weight2)
                        assert (weight1 > 0 and weight2 > 0 and weight1 <= 1 and weight2 <= 1)
                        p_value += weight1 * weight2 * image[l][iy][k]
                proj[i, j] = p_value
        # print('Projection for a loop: ' + str(time.time() - start_time))
        return proj

    @staticmethod
    def _distance_project_on_x(image, CoordY1, CoordY2, CoordZ1, CoordZ2, Yplane, Zplane, image_y1, image_y2, image_z1,
                               image_z2, dy, dz, ix):
        tol_min = 1e-6
        tol_max = 1e7
        proj = np.zeros(CoordY1.shape, dtype=np.float32)
        for i in range(CoordY1.shape[0]):
            for j in range(CoordY1.shape[1]):
                p_value = 0
                s_index_y = min(image_y1[i, j], image_y2[i, j])
                e_index_y = max(image_y1[i, j], image_y2[i, j])
                s_index_z = min(image_z1[i, j], image_z2[i, j])
                e_index_z = max(image_z1[i, j], image_z2[i, j])
                for k in range(int(s_index_y), int(e_index_y) + 1):
                    if (k < 0 or k > image.shape[1] - 1):
                        continue
                    if (s_index_y == e_index_y):
                        weight1 = 1
                    elif (k == s_index_y):
                        weight1 = (max(CoordY1[i, j], CoordY2[i, j]) - Yplane[k + 1]) / abs(
                            CoordY1[i, j] - CoordY2[i, j])
                    elif (k == e_index_y):
                        weight1 = (Yplane[k] - min(CoordY1[i, j], CoordY2[i, j])) / abs(CoordY1[i, j] - CoordY2[i, j])
                    else:
                        weight1 = abs(dy) / abs(CoordY1[i, j] - CoordY2[i, j])
                    # if(abs(weight1) - 0 < tol_min):
                    #    weight1 = 0
                    for l in range(int(s_index_z), int(e_index_z) + 1):
                        if (l < 0 or l > image.shape[2] - 1):
                            continue
                        if (s_index_z == e_index_z):
                            weight2 = 1
                        elif (l == s_index_z):
                            weight2 = (max(CoordZ1[i, j], CoordZ2[i, j]) - Zplane[l + 1]) / abs(
                                CoordZ1[i, j] - CoordZ2[i, j])
                        elif (l == e_index_z):
                            weight2 = (Zplane[l] - min(CoordZ1[i, j], CoordZ2[i, j])) / abs(
                                CoordZ1[i, j] - CoordZ2[i, j])
                        else:
                            weight2 = abs(dz) / abs(CoordZ1[i, j] - CoordZ2[i, j])
                        # print(s_index_z,e_index_z,Zplane[l+1],Zplane[l],CoordZ1[i,j],CoordZ2[i,j])
                        # if(abs(weight2) < tol_min):
                        #    weight2 = 0
                        # print(weight1,weight2)
                        assert (weight1 > 0 and weight2 > 0 and weight1 <= 1 and weight2 <= 1)
                        p_value += weight1 * weight2 * image[l][k][ix]
                proj[i, j] = p_value
        return proj

    @staticmethod
    def _distance_project_on_z(image, CoordX1, CoordX2, CoordY1, CoordY2, Xplane, Yplane, image_x1, image_X2, image_y1,
                               image_y2, dx, dy, iz):
        tol_min = 1e-6
        tol_max = 1e7
        proj = np.zeros(CoordX1.shape, dtype=np.float32)
        for i in range(CoordX1.shape[0]):
            for j in range(CoordX1.shape[1]):
                p_value = 0
                s_index_x = min(image_x1[i, j], image_x2[i, j])
                e_index_x = max(image_x1[i, j], image_x2[i, j])
                s_index_y = min(image_y1[i, j], image_y2[i, j])
                e_index_y = max(image_y1[i, j], image_y2[i, j])
                for k in range(int(s_index_x), int(e_index_x) + 1):
                    if (k < 0 or k > image.shape[0] - 1):
                        continue
                    if (s_index_x == e_index_x):
                        weight1 = 1
                    elif (k == s_index_x):
                        weight1 = (Xplane[k + 1] - max(CoordX1[i, j], CoordX2[i, j])) / abs(
                            CoordX1[i, j] - CoordX2[i, j])
                    elif (k == e_index_x):
                        weight1 = (min(CoordY1[i, j], CoordY2[i, j]) - Xplane[k]) / abs(CoordX1[i, j] - CoordX2[i, j])
                    else:
                        weight1 = abs(dx) / abs(CoordX1[i, j] - CoordX2[i, j])
                    # if(abs(weight1) - 0 < tol_min):
                    #    weight1 = 0
                    for l in range(int(s_index_y), int(e_index_y) + 1):
                        if (l < 0 or l > image.shape[1] - 1):
                            continue
                        if (s_index_z == e_index_z):
                            weight2 = 1
                        elif (l == s_index_y):
                            weight2 = (max(CoordY1[i, j], CoordY2[i, j]) - Yplane[l + 1]) / abs(
                                CoordY1[i, j] - CoordY2[i, j])
                        elif (l == e_index_y):
                            weight2 = (Yplane[l] - min(CoordY1[i, j], CoordY2[i, j])) / abs(
                                CoordY1[i, j] - CoordY2[i, j])
                        else:
                            weight2 = abs(dy) / abs(CoordY1[i, j] - CoordY2[i, j])
                        # print(s_index_z,e_index_z,Zplane[l+1],Zplane[l],CoordZ1[i,j],CoordZ2[i,j])
                        # if(abs(weight2) < tol_min):
                        #    weight2 = 0
                        # print(weight1,weight2)
                        assert (weight1 > 0 and weight2 > 0 and weight1 <= 1 and weight2 <= 1)
                        p_value += weight1 * weight2 * image[iz][l][k]
                proj[i, j] = p_value
        return proj

    def ray(self):
        nViews = self.params['NumberOfViews']
        [nu, nv] = self.params['NumberOfDetectorPixels']
        [dv, du] = self.params['DetectorPixelSize']
        [dx, dy, dz] = self.params['ImagePixelSpacing']
        [nx, ny, nz] = self.params['NumberOfImage']
        Source_Init = np.array(self.params['SourceInit'])
        Detector_Init = np.array(self.params['DetectorInit'])
        StartAngle = self.params['StartAngle']
        EndAngle = self.params['EndAngle']
        Origin = np.array(self.params['Origin'])
        PhantomCenter = np.array(self.params['PhantomCenter'])
        gpu = self.params['GPU']

        SAD = np.sqrt(np.sum((Source_Init - Origin) ** 2))
        SDD = np.sqrt(np.sum((Source_Init - Detector_Init) ** 2))
        angle = np.linspace(StartAngle, EndAngle, nViews + 1)
        angle = theta[0:-1]
        Xplane = (PhantomCenter[0] - (nx - 1) / 2.0 + range(0, nx)) * dx
        Yplane = (PhantomCenter[1] - (ny - 1) / 2.0 + range(0, ny)) * dy
        Zplane = (PhantomCenter[2] - (nz - 1) / 2.0 + range(0, nz)) * dz
        Xplane = Xplane - dx / 2
        Yplane = Yplane - dy / 2
        Zplane = Zplane - dz / 2
        proj = np.zeros([nViews, nu, nv], dtype=np.float32)
        for angle in theta:
            # starting from x-axis and rotating ccw
            SourceX = -SAD * sin(angle)
            SourceY = SAD * cos(angle)
            SourceZ = 0
            DetectorX = (SDD - SAD) * sin(angle)
            DetectorY = -(SDD - SAD) * cos(angle)
            DetectorZ = 0
            DetectorLengthU = range(floor(-nu / 2), floor(nu / 2)) * du
            DetectorLengthV = range(floor(-nv / 2), floor(nv / 2)) * dv
            if (abs(tan(angle)) < tol_min):
                DetectorIndex = [DetectorX + DetectlrLengthU]
                DetectorIndexZ = DetectorZ - DetectorLengthV
            elif (tan(angle) >= tol_max):
                DetectorIndex = [DetectorY + DetectorLengthU]
                DetectorIndexZ = DetectorZ - DetectorLengthV
            else:
                xx = sqrt(DetectorLengthU ** 2 / (1 + tan(angle) ** 2))
                yy = tan(angle) * sqrt(DetectorLengthU ** 2 / (1 + tan(angle) ** 2))
                DetectorIndex = [DetectorX * np.sign(DetectorLengthU * xx), ]
            if (DetectorY > 0):
                DetectorIndex = DetectoIndex[:, ]
            DetectorIndex = DetectorIndex[:, 1:-2]
            DetectorIndexZ = DetectorIndexZ[1:-2]
            if (gpu):
                pass
            else:
                pass

        if (save):
            proj.tofile(write_filename, sep='', format='')

        return proj
