import scipy.io
import numpy as np
import gc
from PIL import Image
from DIP_main import *
from data_reduction import *


def load_3D(fichier, index):
	mat = scipy.io.loadmat(fichier)
	images = mat[index]
	return np.asarray(images)


def getSaveName(directory, x, y, z, denoised = False):
    if denoised:
        return save_directory + "/imgBlock_X" + str(x) + "_Y" + str(y) + "_Z" + str(z) + "_denoised.npy"
    else:
        return save_directory + "/imgBlock_X" + str(x) + "_Y" + str(y) + "_Z" + str(z) + ".npy"


def testsMain(test, num_iter):
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
		save_directory = "tmp_blocks"
		img_np = load_3D('data/18am_T2MS_MCT_norm.mat', "MCT18am_norm")
		img_noisy_np = load_3D('data/18am_T2MS_CBCT_MDL_vert5x5_norm.mat', "CBCTMDLvert5x5_18am_norm")
		final_size = img_noisy_np.shape
		side = 4
		overlap = 15	
		
		img_noisy_np, indexes = crop_image(img_noisy_np, seuil = 0.1, output = True)
		np.save("ground_truth.mat", img_np[indexes[0] : indexes[1], indexes[2] : indexes[3], indexes[4] : indexes[5]])
		np.save("bruite.mat", img_noisy_np)
		
		img_blocks = slide3D(img_noisy_np, side, overlap = overlap)

		for x in range(side):
			for y in range(side):
				for z in range(side):
					np.save(getSaveName(save_directory, x, y, z), img_blocks[x, y, z])
		del img_np
		del img_noisy_np		
		del indexes
		del img_blocks
		gc.collect()

		for x in range(side):
			for y in range(side):
				for z in range(side):            
					current_block = np.load(getSaveName(save_directory, x, y, z))
					print("Denoising : [", str(x), ", ", str(y), ", ", str(z), "] of size", str(current_block.shape))
					current_block, _ = DIP_3D(current_block, num_iter=num_iter, LR=0.005, osirim = False, PLOT=False)
					np.save(getSaveName(save_directory, x, y, z, denoised=True), current_block)
					del current_block
					gc.collect()

		img_blocks_denoised = np.empty((side, side, side), dtype = object)
		for x in range(side):
			for y in range(side):
				for z in range(side):
					fileName = getSaveName(save_directory, x, y, z, denoised=True)
					img_blocks_denoised[x, y, z] = np.load(fileName)

		for fenetrage in ["hamming", "lineaire", "carre"]:
			for moyennage in ["arithmetique", "geometrique", "contreharmonique"]:
				denoised_image = merge3D(img_blocks_denoised, final_size, overlap = overlap, withChannels = True, output = True, fenetrage = fenetrage, moyennage = moyennage)
				np.save("debruite_" + fenetrage + "_" + moyennage + "_" + num_iter + "iter.mat", denoised_image)
				del denoised_image
				gc.collect()


if __name__ == '__main__':
	test = sys.argv[1]
	num_iter = sys.argv[2]
	testsMain(test, num_iter)



