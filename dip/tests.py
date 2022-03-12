import scipy.io
import numpy as np
import gc
from PIL import Image
from DIP_main import *
from data_reduction import *
import matplotlib
import matplotlib.pyplot as plt


def load_3D(fichier, index):
	mat = scipy.io.loadmat(fichier)
	images = mat[index]
	return np.asarray(images)


def getSaveName(directory, x, y, z, denoised = False):
    if denoised:
        return directory + "/imgBlock_X" + str(x) + "_Y" + str(y) + "_Z" + str(z) + "_denoised.npy"
    else:
        return directory + "/imgBlock_X" + str(x) + "_Y" + str(y) + "_Z" + str(z) + ".npy"


def testsMain(test, num_iter, overlap):
    matplotlib.use('Agg')
    plt.ioff()
    if test == 1:
	    img_noisy_np = load_3D('data/shepp_logan.mat', "x")
	    denoised_image, parameters = DIP_3D(img_noisy_np, num_iter=num_iter, LR=0.005, osirim = True, PLOT=False)
    if test == 2:
	    img_np = load_3D('data/18am_T2MS_MCT_norm.mat', "MCT18am_norm")
	    img_noisy_np = load_3D('data/18am_T2MS_CBCT_MDL_vert5x5_norm.mat', "CBCTMDLvert5x5_18am_norm")
	    denoised_image, parameters = DIP_3D(img_noisy_np, img_np=img_np, num_iter=num_iter, LR=0.005, osirim = True, PLOT=False)
    if test == 3:
	    img_np = load_3D('data/37c_T3M1_MCT_norm.mat', "MCT37c_norm")
	    img_noisy_np = load_3D('data/37c_T3M1_CBCT_MDL_vert5x5_norm.mat', "CBCTMDLvert5x5_37c_norm")
	    denoised_image, parameters = DIP_3D(img_noisy_np, img_np=img_np, num_iter=num_iter, LR=0.005, osirim = True, PLOT=False)
    if test == 4:
	    img_np = load_3D('data/18am_T2MS_MCT_norm.mat', "MCT18am_norm")
	    img_noisy_np = load_3D('data/18am_T2MS_CBCT_MDL_vert5x5_norm.mat', "CBCTMDLvert5x5_18am_norm")
	    reduction_test(img_np, img_noisy_np, 4, overlap, num_iter, "18am_tests")
    if test == 5:
	    img_np = load_3D('data/37c_T3M1_MCT_norm.mat', "MCT37c_norm")
	    img_noisy_np = load_3D('data/37c_T3M1_CBCT_MDL_vert5x5_norm.mat', "CBCTMDLvert5x5_37c_norm")
	    reduction_test(img_np, img_noisy_np, 4, overlap, num_iter, "37c_tests")
    if test == 6:
	    img_np = load_3D('data/37c_T3M1_MCT_norm.mat', "MCT37c_norm")
	    img_noisy_np = load_3D('data/37c_T3M1_CBCT_MDL_vert5x5_norm.mat', "CBCTMDLvert5x5_37c_norm")
	    reduction_test(img_np, img_noisy_np, 5, overlap, num_iter, "37c_tests")
    if test == 7:
	    img_np = load_3D('data/37c_T3M1_MCT_norm.mat', "MCT37c_norm")
	    img_noisy_np = load_3D('data/37c_T3M1_CBCT_MDL_vert5x5_norm.mat', "CBCTMDLvert5x5_37c_norm")
	    reduction_test(img_np, img_noisy_np, 6, overlap, num_iter, "37c_tests")
    if test == 8:
	    img_np = load_3D('data/shepp_logan.mat', "x")
	    img_noisy_np = img_np.copy()
	    reduction_test(img_np, img_noisy_np, 2, overlap, num_iter, "shepp_logan64_tests")

		
def reduction_test(img_np, img_noisy_np, side, overlap, num_iter, name):
    name = name + "_" + str(side) + "side_" + str(overlap) + "overlap_" + str(num_iter) + "num_iter"
    if not os.path.isdir(name):
	    os.mkdir(name)
    save_directory = name + "/tmp_blocks"
    if not os.path.isdir(save_directory):
	    os.mkdir(save_directory)
    
    #reduce the size of image
    img_noisy_np, indexes = crop_image(img_noisy_np, seuil = 0.1, output = False)
    final_size = img_noisy_np.shape
    
    #remove the channels dimension if it is only one channel
    if len(final_size) == 4:
	    if final_size[0] == 1:
		    img_np = img_np[0]
		    img_noisy_np = img_noisy_np[0]
		    final_size = img_noisy_np.shape
    
    #save the noised and groundthruth data put on the same dimension
    np.save(name + "/ground_truth.mat", img_np[indexes[0] : indexes[1], indexes[2] : indexes[3], indexes[4] : indexes[5]])
    np.save(name + "/bruite.mat", img_noisy_np)
    
    #cut the image on side*side*side blocks
    img_blocks = slide3D(img_noisy_np, side, overlap = overlap)

    for x in range(side):
	    for y in range(side):
		    for z in range(side):
			    np.save(getSaveName(save_directory, x, y, z), img_blocks[x, y, z])
    
    #remove unnecessary variables to free the more of RAM
    del img_np
    del img_noisy_np		
    del indexes
    del img_blocks
    gc.collect()

    #call DIP on each block
    for x in range(side):
	    for y in range(side):
		    for z in range(side):            
			    current_block = np.load(getSaveName(save_directory, x, y, z))
			    #print("Denoising : [", str(x), ", ", str(y), ", ", str(z), "] of size", str(current_block.shape))			    
			    current_block, _ = DIP_3D(current_block, num_iter=num_iter, LR=0.005, osirim = True, PLOT=False)
			    if current_block.shape[0] == 1:
				    current_block = current_block[0]
			    np.save(getSaveName(save_directory, x, y, z, denoised=True), current_block)
			    del current_block
			    gc.collect()

    #merge the denoised image following different fusion methods
    for fenetrage in ["hamming", "lineaire", "carre"]:
	    for moyennage in ["arithmetique", "geometrique", "contreharmonique"]:
		    
		    #reload the denoized blocks
		    img_blocks_denoised = np.empty((side, side, side), dtype = object)
		    for x in range(side):
			    for y in range(side):
				    for z in range(side):					
					    img_blocks_denoised[x, y, z] = np.load(getSaveName(save_directory, x, y, z, denoised = True))

		    #merge the blocks into a denoised image
		    denoised_image = merge3D(img_blocks_denoised, final_size, overlap = overlap, withChannels = len(final_size) > 3, output = False, fenetrage = fenetrage, moyennage = moyennage)
		    
		    #save the result
		    np.save(name + "/debruite_" + fenetrage + "_" + moyennage + "_" + str(num_iter) + "iter.mat", denoised_image)
		    
		    #remove unnecessary variables to free the more of RAM
		    del img_blocks_denoised
		    del denoised_image
		    gc.collect()


if __name__ == '__main__':
	testsMain(sys.argv[1], sys.argv[2], sys.argv[3])



