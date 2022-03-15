import numpy as np
from matplotlib import pyplot as plt
import math
import scipy.io
import nibabel as nib
from skimage.transform import rescale
import time
from scipy.signal import medfilt
from skimage.metrics import peak_signal_noise_ratio, normalized_root_mse, structural_similarity
from RED3D_DIP3D.utils import *
import argparse
from bm4d import bm4d
import os
import logging
import sys

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

output_file_handler = logging.FileHandler('./output.log')
stdout_handler = logging.StreamHandler(sys.stdout)

logger.addHandler(output_file_handler)
logger.addHandler(stdout_handler)

def prepare_data(image_path = './shepp_logan_128.mat', crop = False,
                 psf_sz = 9, gaussian_std = 3,  BSNRdb = 10 , input_path="./input/"):
    logger.debug("Running file : " + image_path)
    mat = scipy.io.loadmat(image_path)
    x = mat[list(mat.keys())[-1]]
    if crop:
        x, d0, d1, d2 = crop_image(x)

    # Size of HR image
    mh, nh, th = x.shape

    scale = 2
    # The decimation rates
    dr = scale;
    dc = scale;
    ds = scale
    d = dr * dc * ds

    ml = int(mh / dr)
    nl = int(nh / dc)
    tl = int(th / ds)
    m = ml * nl * tl

    blksz = [ml, nl, tl]  # size of the LR image

    # filter size and std of the Gaussian filter
    # psf_sz = 9
    # gaussian_std = 3
    # create gaussian filter
    H = fspecial3('gaussian', psf_sz, gaussian_std)

    # we pad the blurring kernel and blur the image in the Fourier domain
    # we pad the blurring kernel

    m0, n0, t0 = H.shape
    pads1, pads2 = np.floor([(mh - m0 + 1) / 2, (nh - n0 + 1) / 2, (th - t0 + 1) / 2]).astype(int), np.around(
        [(mh - m0 - 1) / 2, (nh - n0 - 1) / 2, (th - t0 - 1) / 2]).astype(int)
    hpad = np.pad(H, ((pads1[0], pads2[0]), (pads1[1], pads2[1]), (pads1[2], pads2[2])))
    hpad = np.asarray(hpad)
    # Centre the dc of the padded image
    hp_c = np.fft.fftshift(hpad)
    hp_c = np.asarray(hp_c)
    # Fourier transform of the padded blurring kernel
    FB = np.fft.fftn(hp_c)
    # Conjugate of the fourier transform of the padded blurring kernel
    FBC = np.conjugate(FB)
    # Crash in here
    F2B = np.absolute(FB) ** 2
    # print("END1")

    # we blur the image in in the fourier domain.
    # Hx = np.real(np.fft.ifftn(np.matmul(np.asarray(FB), x)))
    # Hx = np.real(np.fft.ifftn(np.multiply(FB,np.fft.fftn(x))))
    Hx = np.real(np.fft.ifftn(FB * np.fft.fftn(x)))
    # print(Hx.shape)

    # We decimate the blurred image
    SHx = Hx[0:Hx.shape[0]:dr, 0:Hx.shape[1]:dc, 0:Hx.shape[2]:ds]

    # We add noise
    N = SHx.shape[0] * SHx.shape[1] * SHx.shape[2]
    # BSNRdb = 10

    noise_sigma = fro(SHx - np.mean(SHx)) / math.sqrt(N * 10 ** (BSNRdb / 10))
    noise = noise_sigma * np.random.normal(size=SHx.shape)

    noise_var = noise_sigma ** 2

    # print(noise_var)
    SHx_n = SHx + noise

    # Save input
    if not os.path.exists(input_path):
        os.makedirs(input_path)

    ## Numpy
    with open(os.path.join(input_path, "degraded_image.npy"), 'wb') as f:
        np.save(f, SHx_n)
    f.close()

    ## Matlab
    mdic = {"label": "Degraded image of RED BM4D", "input": SHx_n}
    scipy.io.savemat(os.path.join(input_path, "degraded_image.mat"), mdic)

    ## .nii.gz
    new_image = nib.Nifti1Image(SHx_n, affine=np.eye(4))
    new_image.header.get_xyzt_units()
    new_image.to_filename(os.path.join(input_path, "degraded_image.nii.gz"))

    ## We upsample using the idea of conjugate transpose
    DTy = np.zeros((mh, nh, th))
    DTy[0:mh:dr, 0:nh:dc, 0:th:ds] = SHx_n

    x_est = rescale(SHx_n, scale, preserve_range=True, anti_aliasing=True)
    psnr_in = peak_signal_noise_ratio(x, x_est)
    logger.debug('PSNR in = {:.3f}'.format(psnr_in))

    return x, x_est, noise_var, FBC, DTy, FB, blksz, F2B, d

def runfp(x, x_est, noise_var, FBC, DTy, FB, blksz, F2B, d ,max_iter = 101, lam = None, sigma = 0.08,
          useparallel=False, input_path ="./input/", output_path="./output", save_output = None):
    sumLoop = 0
    sumLoop_part1 = 0
    sumLoop_part2 = 0
    sumLoop_part3 = 0

    # Save input
    if not os.path.exists(input_path):
        os.makedirs(input_path)

    # Save output
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    ## Numpy
    with open(os.path.join(input_path,"input.npy"), 'wb') as f:
        np.save(f, x_est)
    f.close()

    ## Matlab
    mdic = {"label": "Input of RED BM4D", "input": x_est}
    scipy.io.savemat(os.path.join(input_path,"input.mat"), mdic)

    ## .nii.gz
    new_image = nib.Nifti1Image(x_est, affine=np.eye(4))
    new_image.header.get_xyzt_units()
    new_image.to_filename(os.path.join(input_path,"input.nii.gz"))

    if lam == None:
        lam = 1e1 * noise_var
    psnr_final = 0
    logger.debug('iter\t PSNR \tNRMSE \tSSIM \tTotal_time \tTime_Denoiser \tTime_FFT \tTime_InverseFFT\n')
    for i in range(max_iter):
        timeLoop = time.time()
        # Apply the denoiser
        timeLoop_part1 = time.time()
        # f_x_est = medfilt(x_est)
        # lmax = 3
        # basis = 'haar'
        # mu = thrextr_3d(x_est,basis)
        # f_x_est = ST3Dwt(x_est, basis, lmax, mu)

        ## FastDVDNet
        # x_est_new = [x_est, x_est, x_est]
        # x_est_new = np.stack(x_est_new, axis=3)
        # f_x_est = test_fastdvdnet(argspar, x_est_new.astype(np.float32)).cpu().numpy()
        # f_x_est = np.reshape(f_x_est, (f_x_est.shape[0], f_x_est.shape[2], f_x_est.shape[3], 3))
        # f_x_est = f_x_est[:, :, :, 1]

        #BM4D
        f_x_est = bm4d(x_est,sigma)

        sumLoop_part1 += time.time() - timeLoop_part1

        # Determine FFT of k
        timeLoop_part2 = time.time()
        FR1 = FBC * np.fft.fftn(DTy) + np.fft.fftn(lam * f_x_est)
        F2D = 1

        # Entrywise product

        x11 = FB * FR1 / F2D
        FBR1 = blcthre3d(x11, blksz)
        invW1 = blcthre3d(F2B / F2D, blksz)
        invWBR1 = FBR1 / (invW1 + lam * d)
        sumLoop_part2 += time.time() - timeLoop_part2

        # Inverse FFT and subtract from k
        timeLoop_part3 = time.time()
        FCBinvWBR1 = blockproc3(FBC, blksz, invWBR1, useparallel=useparallel)
        FX1 = (FR1 - FCBinvWBR1) / F2D / lam
        x_est = np.real(np.fft.ifftn(FX1))
        sumLoop_part3 += time.time() - timeLoop_part3

        TL = time.time() - timeLoop
        sumLoop = sumLoop + TL
        if i % 10 == 0 or True :
            im_out = x_est
            psnr_out = peak_signal_noise_ratio(x, im_out)
            nrmse_out = normalized_root_mse(x, im_out)
            ssim_out = structural_similarity(x, im_out)
            if psnr_final < psnr_out:
                x_est_final = x_est
                psnr_final = psnr_out
            logger.debug('{}\t {:.3f}\t {:.3f}\t {:.3f}\t {:.3f}\t {:.3f}\t {:.3f}\t {:.3f}'.format(i, psnr_out, nrmse_out, ssim_out,
                                                                                         sumLoop, sumLoop_part1, sumLoop_part2, sumLoop_part3))
            if save_output != None and i%save_output == 0:
                ## Matlab
                mdic = {"label": "Output of RED BM4D", "output": im_out}
                scipy.io.savemat(os.path.join(output_path, "output_" + str(i) + ".mat"), mdic)

    im_out = x_est_final
    psnr_out = peak_signal_noise_ratio(x, im_out)
    nrmse_out = normalized_root_mse(x, im_out)
    ssim_out = structural_similarity(x, im_out)
    logger.debug("RED-FP method : PSNR : {:.3f}dB - NRMSE : {:.3f} - SSIM : {:.3f} ".format(psnr_out,nrmse_out,ssim_out))
    ## Numpy
    with open(os.path.join(output_path,"output_fin.npy"), 'wb') as f:
        np.save(f, im_out)
    f.close()
    ## Matlab
    mdic = {"label": "Output of RED BM4D", "output": im_out}
    scipy.io.savemat(os.path.join(output_path,"output_fin.mat"),mdic)

    ## .nii.gz
    new_image = nib.Nifti1Image(im_out, affine=np.eye(4))
    new_image.header.get_xyzt_units()
    new_image.to_filename(os.path.join(output_path,"output_fin.nii.gz"))

    return im_out, psnr_out

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description="Denoise a image 3D with BM4D and Regularization by Denoising (RED)")
    parser.add_argument("--image_path", type=str,
                        default="./shepp_logan_256.mat",
                        help='path to load original image')
    parser.add_argument("--input_path", type=str,
                        default="./input/",
                        help='folder to save input of image (3 formats : npy, mat,nii.gz)')
    parser.add_argument("--output_path", type=str,
                        default="./output/",
                        help='folder to save output of image (3 formats : npy, mat,nii.gz)')
    parser.add_argument("--crop", action='store_true',
                        help='reduce the size of image')
    parser.add_argument("--psf_sz", type=int, default=9, help='size of filter')
    parser.add_argument("--gaussian_std", type=float, default=3, help='gaussian_std')
    parser.add_argument("--BSNRdb", type=int, default=10, help='noise level used on image')
    parser.add_argument("--max_iter", type=int, default=101, help='number iteration of fix-point method')
    parser.add_argument("--lam", type=float, default=0.02, help='params lambda of fix-point method')
    parser.add_argument("--sigma", type=float, default=0.1,help='params sigma of BM4D method')
    parser.add_argument("--use_parallel", action='store_true',
                        help='use multiprocessing with RED')
    parser.add_argument("--save_output", type=int, default = None,
                        help='Save output for each x iterations')

    argspar = parser.parse_args()

    logger.debug(argspar)

    x, x_est, noise_var, FBC, DTy, FB, blksz, F2B, d = prepare_data(argspar.image_path, argspar.crop,
                                                                    argspar.psf_sz, argspar.gaussian_std,
                                                                    argspar.BSNRdb, argspar.input_path)

    im_out, psnr_out = runfp(x, x_est, noise_var, FBC, DTy, FB, blksz, F2B, d,
                             argspar.max_iter, argspar.lam, argspar.sigma, argspar.use_parallel, argspar.input_path, argspar.output_path,  argspar.save_output)

    logger.debug("---------------- END FILE ----------------\n")