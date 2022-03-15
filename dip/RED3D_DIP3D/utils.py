import numpy as np
from matplotlib import pyplot as plt
import math
import scipy.io
import nibabel as nib
from skimage.transform import rescale
import time
from scipy.signal import medfilt
import multiprocessing
import pywt


def fspecial3(mode, size, sigma=None):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    if mode == "average":
        g = np.empty((size, size, size))
        g.fill(size * size * size)
        g = 1 / g
    elif mode == "gaussian":
        x, y, z = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
        g = np.exp(-((x ** 2 + y ** 2 + z ** 2) / (2.0 * sigma ** 2)))
        g = g / g.sum()
    return g


def fro(aa):
    # This function gives the Frobenius norm of a 3D volume
    result = aa ** 2
    return np.sqrt(np.sum(result))


def PSNR(orig_im, est_im):
    d1 = np.max(orig_im)
    d2 = np.max(est_im)
    d = d1 if d1 > d2 else d2
    mse = np.sum((orig_im - est_im) ** 2) / (np.prod(orig_im.shape))
    #     print(d,mse)
    PSNR = 10 * np.log10(d ** 2 / mse)
    return PSNR


def fd(mat, nR, nC, nS):
    return ipermute(np.reshape(mat, [nR, nC, nS], order='F'), [0, 1, 2])


def ipermute(mat, order):
    inverseorder = np.zeros(len(order), dtype='int')
    inverseorder[order] = [i for i in range(len(order))]
    return permute(mat, inverseorder)


def permute(mat, order):
    return np.transpose(mat, order)


def ufd_1(A, R, C):  # /!\ : les dimensions (R et C) commencent à 0 en python et non à 1 comme dans Matlab !
    I = A.shape
    J = np.prod([I[r] for r in R])
    K = np.prod([I[c] for c in C])
    L = np.transpose(A, np.append(R, C))
    return L.reshape([J, K], order='F')


def blcthre3d(im, blksz):
    if len(blksz) == 0:
        blksz = im.shape

    # image size
    imsz = im.shape

    # starting points
    sr = np.array([x for x in range(0, imsz[0], blksz[0])])
    sc = np.array([x for x in range(0, imsz[1], blksz[1])])
    ss = np.array([x for x in range(0, imsz[2], blksz[2])])

    # end points
    er = sr + blksz[0] - 1
    ec = sc + blksz[1] - 1
    es = ss + blksz[2] - 1

    #     print(sr,er)
    #     print(sc,ec)
    #     print(ss,es)

    # number of blocks in each dimension
    nbr = len(sr)
    nbc = len(sc)
    nbs = len(ss)

    # compute size of vectorized block
    szbl = blksz[0] * blksz[1] * blksz[2]

    # vector matrix
    #     print(szbl,(imsz[0]/blksz[1])*(imsz[1]/blksz[1])*(imsz[2]/blksz[2]))
    blocvec = np.zeros((szbl, int((imsz[0] / blksz[0]) * (imsz[1] / blksz[1]) * (imsz[2] / blksz[2]))),
                       dtype=np.complex128)

    #     print(blocvec.shape)
    # fun = lambda ufd_1, x, m : ufd_1(x, [1], [2, 3]).reshape(m, 1)
    index = 0
    # iterate all image blocks
    for i in range(nbr):
        for j in range(nbc):
            for k in range(nbs):
                #                 print(i,j,k)
                # process current image block
                #                 ufd_1(A=im[sr[i]:er[i]+1, sc[j]:ec[j]+1, ss[k]:es[k]+1],R = [0],C=[1,2])
                prblk = ufd_1(A=im[sr[i]:er[i] + 1, sc[j]:ec[j] + 1, ss[k]:es[k] + 1], R=[0], C=[1, 2]).reshape(
                    [szbl, 1], order='F')
                #                 prblk = fun(im[sr[i]:er[i]+1, sc[j]:ec[j]+1, ss[k]:es[k]+1])
                #                 np.squeeze(prblk)
                #                 print(type(prblk[0,0]))
                blocvec[:, index] = np.squeeze(prblk)
                index = index + 1
    blocsin = np.sum(blocvec, axis=1)
    #     print(blocsin)
    blocresult = fd(blocsin, blksz[0], blksz[1], blksz[2])

    return blocresult


def multiply(x, invWBR1):
    return x * invWBR1


def blockproc3(im, blksz, invWBR1, border=[0, 0, 0], useparallel=False):
    """
    BLOCKPROC3  Block processing for a 3D image
    Sometimes images are too large to be processed in one block. Matlab
    provides function blockproc(), that works on 2D images only. This
    blockproc3() function, on the other hand, works also with 3D images.
    The original Matlab function blockproc() can run processing in parallel
    (if the ). This function doesn't
    support that functionality yet.
    function IM2 = blockproc3(im, blksz, fun)
    IM is a 2D or 3D-array with a grayscale image.
    BLKSZ is a 2- or 3-vector with the size of the blocks the image will be
    split into. The last block in every row, column and slice will be
    smaller, if the image cannot accommodate it.
    FUN is a function handle. This is the processing applied to every
    block.
    IM2 = blockproc3(IM, BLKSZ, FUN, BORDER)
    BORDER is a 2- or 3-vector with the size of the border around each
    block. Basically, each block size will be grown by this amount, the
    block processed, and then the border cut off from the solution. This is
    useful to avoid "seams" in IM2.
    Example:
    fun = @(x) deconvblind(x, ones(20, 20, 10));
    im2 = blockproc3d(im, [256 256 128], fun, [30 30 10]);
    IM2 = blockproc3(..., PARALLEL)
    PARALLEL is a boolean flag to activate parallel processing if the
    Parallel Computing Toolbox is available, and a worker pool has been
    opened. When PARALLEL=true, all workers will process blocks in
    parallel (default: PARALLEL=false).
    Parallel processing works by first splitting the image into blocks and
    storing them in a cell array, then processing each block, and finally
    reassembling the cell array into the output image.
    This will increase the amount of necessary memory. Speed improvements
    may not be as good as expected, as Matlab function can be already more
    or less optimised to make use of several processors.
    To use this option, first it's necessary to create a pool of workers in
    Matlab. You can find more information in Matlab's documentation (e.g.
    help matlabpool). A simple example:
    median filtering using a 19x19x19 voxel neighbourhood, processing
    the image by blocks, using parallel computations
    matlabpool open
    sz = [19 19 19];
    fun = @(x) medfilt3(x, sz);
    im2 = blockproc3(im, [128 128 64], fun, (sz+1)/2, true);
    matlabpool close
    See also: blockproc.

    Author: Ramon Casero <rcasero@gmail.com>
    Copyright Â? 2011 University of Oxford
    Version: 0.2.0
    $Rev: 772 $
    $Date: 2012-04-10 23:53:41 +0200 (Tue, 10 Apr 2012) $
    University of Oxford means the Chancellor, Masters and Scholars of
    the University of Oxford, having an administrative office at
    Wellington Square, Oxford OX1 2JD, UK.
    This file is part of Gerardus.
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details. The offer of this
    program under the terms of the License is subject to the License
    being interpreted in accordance with English Law and subject to any
    action against the University of Oxford being under the jurisdiction
    of the English Courts.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    """
    if len(blksz) == 0:
        blksz = im.shape

    # for convenience, we need the size vector to have 3 components, even for a 2D image
    if len(blksz) < 3:
        blksz[2] = 1

    imsz = im.shape

    # starting points
    r0 = np.array([x for x in range(0, imsz[0], blksz[0])])
    c0 = np.array([x for x in range(0, imsz[1], blksz[1])])
    s0 = np.array([x for x in range(0, imsz[2], blksz[2])])

    # end points
    rx = r0 + blksz[0] - 1
    cx = c0 + blksz[1] - 1
    sx = s0 + blksz[2] - 1
    rx = np.array([min(x, imsz[0]) for x in rx])
    cx = np.array([min(x, imsz[1]) for x in cx])
    sx = np.array([min(x, imsz[2]) for x in sx])

    # block limits with the extra borders
    # starting points
    br0 = np.array([max(x - border[0], 0) for x in r0])
    bc0 = np.array([max(x - border[1], 0) for x in c0])
    bs0 = np.array([max(x - border[2], 0) for x in s0])

    # end points
    brx = np.array([min(x + border[0], imsz[0]) for x in rx])
    bcx = np.array([min(x + border[1], imsz[1]) for x in cx])
    bsx = np.array([min(x + border[2], imsz[2]) for x in sx])

    # number of blocks in each dimension
    NR = len(r0)
    NC = len(c0)
    NS = len(s0)

    #     print(r0,rx,br0,brx)

    # init output
    im2 = np.zeros_like(im)
    #     im2=im

    # parallel procesing
    if useparallel:
        # number of blocks
        numblocks = NR * NC * NS

        # generate all input blocks (loops are in inverted order, so that linear indices follow 1, 2, 3, 4...)
        list_params = []
        blocks = []
        for i in range(NR):
            for j in range(NC):
                for k in range(NS):
                    #                     idx = sub2ind([NR, NC, NS], i, j, k)
                    #                     blocks[idx] = im[br0[i]:brx[i]+1, bc0[j]:bcx[j]+1, bs0[k]:bsx[k]+1]
                    list_params.append([im[br0[i]:brx[i] + 1, bc0[j]:bcx[j] + 1, bs0[k]:bsx[k] + 1], invWBR1])

        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        blocks = pool.starmap(multiply, list_params)
        #         im2 = pool.starmap(multiply, range(numblocks))
        pool.close()
        pool.join()

        for b in range(numblocks):
            # block's array indices from linear indices
            #           [i, j, k] = ind2sub([len(br0), len(bc0), len(bs0)], b)
            [i, j, k] = np.unravel_index(b, (len(br0), len(bc0), len(bs0)))
            #             print(i, j, k)
            # assign result to output removing the borders
            im2[r0[i]:rx[i] + 1, c0[j]:cx[j] + 1, s0[k]:sx[k] + 1] = blocks[b][r0[i] - br0[i]:(rx[i] - br0[i] + 1),
                                                                     c0[j] - bc0[j]:(cx[j] - bc0[j] + 1),
                                                                     s0[k] - bs0[k]:(sx[k] - bs0[k] + 1)]
    # single processor (we save memory by not creating a cell vector with all the blocks)                                                               s0[k]-bs0[k]+1:sx[k]-bs0[k]+1]
    else:
        for i in range(NR):
            for j in range(NC):
                for k in range(NS):
                    aux = multiply(im[br0[i]:brx[i] + 1, bc0[j]:bcx[j] + 1, bs0[k]:bsx[k] + 1], invWBR1)
                    #                     print(i,j,k)
                    # assign result to output removing the borders
                    #                     print(im2[r0[i]:rx[i]+1, c0[j]:cx[j]+1, s0[k]:sx[k]+1].shape)
                    #                     print(aux[r0[i]-br0[i]+1:(brx[i]-rx[i])+1,c0[j]-bc0[j]+1:(bcx[j]-cx[j])+1,s0[k]-bs0[k]+1:(bsx[k]-sx[k])+1].shape)
                    im2[r0[i]:rx[i] + 1, c0[j]:cx[j] + 1, s0[k]:sx[k] + 1] = aux[r0[i] - br0[i]:(
                                aux.shape[0] - (brx[i] - rx[i])),
                                                                             c0[j] - bc0[j]:(aux.shape[1] - (
                                                                                         bcx[j] - cx[j])),
                                                                             s0[k] - bs0[k]:(aux.shape[2] - (
                                                                                         bsx[k] - sx[k]))]

    return im2


def ST3Dwt(img, basis, lmax, mu):
    imgw = {}
    approx = {}
    for i in range(lmax):
        if (i == 0):
            imgw.setdefault(i, pywt.dwtn(img, basis))
            #             print(imgw.keys())
            approx.setdefault(i, imgw[i]['aaa'])
        else:
            imgw.setdefault(i, pywt.dwtn(approx[i - 1], basis))
            approx.setdefault(i, imgw[i]['aaa'])

        for k in imgw[i].keys():
            # soft thresholding
            temp = imgw[i][k]
            #             print("1: ", temp.shape)
            temp = pywt.threshold(temp, mu, 'soft')
            #             print("2: ", temp.shape)
            imgw[i][k] = temp

        mu = mu / np.log(((i + 1) ** 2) + 1)
    for i in reversed(range(lmax)):
        if (i > 0):
            imgw[i - 1]['aaa'] = pywt.idwtn(imgw[i], basis)
        else:
            res = pywt.idwtn(imgw[i], basis)

    return res


def thrextr_3d(im, wnam):
    imgw = pywt.dwtn(im, wnam)
    N = np.prod(im.shape)
    det_coeff = []
    #     print(imgw.keys())
    for k in imgw.keys():
        if k == 'aaa':
            continue
        #         print(k)
        temp = imgw[k]
        det_coeff.append(temp)

    noi_var = np.median(np.abs(det_coeff)) / 0.6745

    T = noi_var * np.sqrt(2 * np.log(N * np.log2(N)))
    return T

def crop_image(x):
  dimension_0 = []
  for i in range(x.shape[0]):
    if np.all(x[i,:,:] == 0) == False:
    # if np.sum(x[i,:,:]) > 11000:
        dimension_0.append(i)
  if len(dimension_0) %2 == 1:
    x = x[dimension_0[1:],:,:]
    dimension_0 = dimension_0[1:]
  else:
    x = x[dimension_0,:,:]

  dimension_1 = []
  for i in range(x.shape[1]):
    if np.all(x[:,i,:] == 0) == False:
    # if np.sum(x[:,i,:]) > 11000:
        dimension_1.append(i)
  if len(dimension_1) %2 == 1:
    x = x[:,dimension_1[1:],:]
    dimension_1 = dimension_1[1:]
  else:
    x = x[:,dimension_1,:]

  dimension_2 = []
  for i in range(x.shape[2]):
    if np.all(x[:,:,i] == 0) == False:
    # if np.sum(x[:,:,i]) > 11000:
        dimension_2.append(i)
  if len(dimension_2) %2 == 1:
    x = x[:,:,dimension_2[1:]]
    dimension_2 = dimension_2[1:]
  else:
    x = x[:,:,dimension_2]
  return x,dimension_0,dimension_1,dimension_2