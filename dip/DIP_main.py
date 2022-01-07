
#Import libs

from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cm

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from models.skip import *

import torch
import torch.optim

from skimage.metrics import peak_signal_noise_ratio
from utils.denoising_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor


#Tool function

def get_noisy_image(img_np, sigma):
    """Adds Gaussian noise to an image.
    
    Args: 
        img_np: image, np.array with values from 0 to 1
        sigma: std of the noise
    """
    img_noisy_np = np.clip(img_np + np.random.normal(scale=sigma, size=img_np.shape), 0, 1).astype(np.float32)
    img_noisy_pil = np_to_pil(img_noisy_np)

    return img_noisy_pil, img_noisy_np
  

def Setup(img_pil, img_noisy_np, INPUT = 'noise', pad = 'reflection', input_depth = 1):

  #INPUT = 'noise' 'meshgrid'
  #OPTIMIZER='adam'  'LBFGS'

  reg_noise_std = 1./30. #set to 1./20. for sigma=50
  net = skip(
              input_depth, 1, 
              num_channels_down = [128, 128, 128, 128, 128], 
              num_channels_up   = [128, 128, 128, 128, 128],
              num_channels_skip = [4, 4, 4, 4, 4], 
              upsample_mode='bilinear',
              need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')
  net = net.type(dtype)      
  net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()
  mse = torch.nn.MSELoss().type(dtype) #Loss
  img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)
  return net, net_input, img_noisy_torch, mse

def closure(params):#reg_noise_std, net_input_saved, noise, net
  if params['reg_noise_std'] > 0:
      params['net_input'] = params['net_input_saved'] + params['noise'].normal_() * params['reg_noise_std']
  out = params['net'](params['net_input'])
  out_avg = None

  # Smoothing
  if params['out_avg'] is None:
      params['out_avg'] = out.detach()
  else:
      params['out_avg'] = params['out_avg'] * params['exp_weight'] + out.detach() * (1 - params['exp_weight'])
          
  total_loss = params['mse'](out, params['img_noisy_torch'])
  total_loss.backward()

  psrn_noisy = peak_signal_noise_ratio(params['img_noisy_np'], out.detach().cpu().numpy()[0][0]) 
  psrn_gt    = peak_signal_noise_ratio(params['ar'], out.detach().cpu().numpy()[0]) 
  psrn_gt_sm = peak_signal_noise_ratio(params['ar'], params['out_avg'].detach().cpu().numpy()[0]) 
  
  # Note that we do not have GT for the "snail" example
  # So 'PSRN_gt', 'PSNR_gt_sm' make no sense
  print ('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f' % (params['i'], total_loss.item(), psrn_noisy, psrn_gt, psrn_gt_sm), '\r', end='')
  if  params['PLOT'] and params['i'] % params['show_every'] == 0:
      out_np = torch_to_np(out)
      plot_image_grid([np.clip(out_np, 0, 1), np.clip(torch_to_np(params['out_avg']), 0, 1)], factor=params['figsize'], nrow=1)
      print("noisy / output : ", psrn_noisy, "initial / output : ", psrn_gt, "initial / avg_output : " , psrn_gt_sm)

  # Backtracking
  if params['i'] % params['show_every']:
      if psrn_noisy - params['psrn_noisy_last'] < -5: 
          print('Falling back to previous checkpoint.')
          for new_param, net_param in zip(last_net, params['net'].parameters()):
              net_param.data.copy_(new_param.cuda())
          return total_loss*0
      else:
          last_net = [x.detach().cpu() for x in params['net'].parameters()]
          params['psrn_noisy_last'] = psrn_noisy
          
  params['i'] += 1
  return total_loss
  

def DIP_2D(img_np): #Main function
  #Set parameters  
  imsize =-1
  PLOT = True
  sigma = 6
  sigma_ = sigma/255.
  OPT_OVER = 'net'
  LR = 0.01
  OPTIMIZER='adam'
  show_every = 10
  exp_weight=0.99
  num_iter = 500
  figsize = 5 

  # Convert image
  im = Image.fromarray((img_np * 255).astype(np.uint8)[:,:,32])

  # Add synthetic noise
  img_pil = crop_image(im, d=32)
  ar = np.array(img_pil)[None, ...]
  ar = ar.astype(np.float32) / 255
      
  img_noisy_pil, img_noisy_np = get_noisy_image(ar, sigma_)
  if PLOT:
      plot_image_grid([ar, img_noisy_np], 4, 6);

  # Setup the DIP
  net, net_input, img_noisy_torch, mse = Setup(img_pil, img_noisy_np)
  p = get_params(OPT_OVER, net, net_input)

  closure_params = {}
  closure_params['reg_noise_std'] = 1./30. #set to 1./20. for sigma=50
  closure_params['net_input_saved'] = net_input.detach().clone()
  closure_params['noise'] = net_input.detach().clone()
  closure_params['net'] = net
  closure_params['out_avg'] = None
  closure_params['mse'] = mse
  closure_params['psrn_noisy_last'] = 0
  closure_params['i'] = 0
  closure_params['img_noisy_torch'] = img_noisy_torch
  closure_params['img_noisy_np'] = img_noisy_np
  closure_params['net_input'] = net_input
  closure_params['ar'] = ar
  closure_params['PLOT'] = PLOT
  closure_params['show_every'] = show_every
  closure_params['figsize'] = figsize
  closure_params['exp_weight'] = exp_weight
  closure_initialized = lambda : closure(closure_params)

  # Optimize
  optimize(OPTIMIZER, p, closure_initialized, LR, num_iter)

  #Display final results
  out_np = torch_to_np(net(net_input))
  q = plot_image_grid([np.clip(out_np, 0, 1), ar, img_noisy_np], factor=13);
  
  return out_np, net.parameters()
