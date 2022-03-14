import gc
import numpy as np


#réduit si possible la taille des données
def crop_image(img, seuil = 0, forceEven = False):
    x_size, y_size, z_size = img.shape
    print("image originale : x=", x_size, "y=", y_size, "z=", z_size)
    
    first_X = x_size
    first_Y = y_size
    first_Z = z_size
    last_X = 0
    last_Y = 0
    last_Z = 0

    for x in range(x_size):
        for y in range(y_size):
            for z in range(z_size):
                if(img[x, y, z] > seuil):
                    if x + 1 > last_X:
                        last_X = x + 1
                    if y + 1 > last_Y:
                        last_Y = y + 1
                    if z + 1 > last_Z:
                        last_Z = z + 1
                    if x < first_X:
                        first_X = x
                    if y < first_Y:
                        first_Y = y
                    if z < first_Z:
                        first_Z = z

    if forceEven:
        first_X, last_X = forceEvenDistance(first_X, last_X, 0, x_size)
        first_Y, last_Y = forceEvenDistance(first_Y, last_Y, 0, y_size)
        first_Z, last_Z = forceEvenDistance(first_Z, last_Z, 0, z_size)

    img = img[first_X : last_X, first_Y : last_Y, first_Z : last_Z]
    x_size, y_size, z_size = img.shape
    print("image rognée : x=", x_size, "y=", y_size, "z=", z_size)
    return img, (first_X, last_X, first_Y, last_Y, first_Z, last_Z)


def forceEvenDistance(firstValue, lastValue, minValue, maxValue):
    if (lastValue - firstValue) % 2 == 0:
      return firstValue, lastValue
    else:
      if lastValue < maxValue:
        return firstValue, lastValue + 1
      else:
        if firstValue > minValue:
          return firstValue - 1, lastValue
        else:
          return firstValue, lastValue - 1
    
    
def slide3D(img, side, overlap = 0):

    size_x, size_y , size_z = img.shape
    sliced_img = np.empty((side, side, side), dtype = object)
    x_step = size_x // side
    y_step = size_y // side
    z_step = size_z // side

    x_remain = size_x % side    
    x_start = 0

    for x in range(side):

        x_stop = x_start + x_step + int(x_remain > 0)
        if not x == 0:
            x_stop += overlap
        if not x == side - 1:
            x_stop += overlap
        y_start = 0
        y_remain = size_y % side
        
        for y in range(side):

            y_stop = y_start + y_step + int(y_remain > 0)
            if not y == 0:
                y_stop += overlap
            if not y == side - 1:
                y_stop += overlap
            z_start = 0
            z_remain = size_z % side

            for z in range(side):
   
                z_stop = z_start + z_step + int(z_remain > 0)
                if not z == 0:
                    z_stop += overlap
                if not z == side - 1:
                    z_stop += overlap
                
                sliced_img[x, y, z] = img[x_start : x_stop, y_start : y_stop, z_start : z_stop]

                z_start = z_stop -  2 * overlap
                z_remain -= 1

            y_start = y_stop - 2 * overlap
            y_remain -= 1

        x_start = x_stop - 2 * overlap
        x_remain -= 1

    return sliced_img


def meanCustom(values, methode = "arithmetique"):
    
    valuesToMean = np.zeros((len(values)), dtype = float)
    coefs = np.zeros((len(values)), dtype = float)
    for index, pair in enumerate(values):
        valuesToMean[index] = pair[0]
        coefs[index] = pair[1]

    if methode == "geometrique":
        return np.prod(valuesToMean ** coefs) ** (1 / sum(coefs)) #moyenne géométrique
    elif methode == "contreharmonique":
        return sum(np.square(valuesToMean) * coefs) / sum(valuesToMean * coefs) #moyenne contreharmonique
    else:
        return sum(valuesToMean * coefs) / sum(coefs) #moyenne arithmétique
    

def merge3D(denoised_blocks, final_size, overlap = 0, withChannels = False, fenetrage = "hamming", moyennage = "arithmetique", output = True):
    x_size, y_size, z_size = denoised_blocks.shape
    if withChannels :
        values = np.empty((1, int(final_size[0]), int(final_size[1]), int(final_size[2])), dtype=object)
    else:
        values = np.empty((int(final_size[0]), int(final_size[1]), int(final_size[2])), dtype=object)
    
    if fenetrage == "lineaire": #fenetre linéaire
        fenetre = np.ones((overlap, 1)) / overlap
        for i in range(overlap):
            fenetre[i] = fenetre[i] * i
    elif fenetrage == "carre": #fenetre au carré
        fenetre = np.ones((overlap, 1)) / overlap
        for i in range(overlap): 
            fenetre[i] = (fenetre[i] * i) ** 2
    else: #fenetre de hamming
        fenetre = np.hamming(2*overlap)[:overlap]

    blocks_sizes_x = np.ones(x_size, dtype=int) * int(final_size[0] // x_size)
    blocks_rest_x = int(final_size[0] % x_size)
    blocks_sizes_y = np.ones(y_size, dtype=int) * int(final_size[1] // y_size)
    blocks_rest_y = int(final_size[1] % y_size)
    blocks_sizes_z = np.ones(z_size, dtype=int) * int(final_size[2] // z_size)
    blocks_rest_z = int(final_size[2] % z_size)

    for i in range(final_size[0]):
      if blocks_rest_x > 0:
          blocks_rest_x -= 1
          blocks_sizes_x[i] += 1
      else :
          break
    for i in range(final_size[1]):
      if blocks_rest_y > 0:
          blocks_rest_y -= 1
          blocks_sizes_y[i] += 1
      else :
          break
    for i in range(final_size[2]):
      if blocks_rest_z > 0:
          blocks_rest_z -= 1
          blocks_sizes_z[i] += 1
      else :
          break

    start_write_x = 0
    begin_i = 0                
    for x in range(x_size):
        start_write_y = 0
        begin_j = 0
        for y in range(y_size):                    
            start_write_z = 0
            begin_k = 0
            for z in range(z_size):
                if output:
                    print("Merging block localized in [" + str(x) + ", " + str(y) + ", " + str(z) + "]")
                if withChannels :
                    current_shape = denoised_blocks[x, y, z][0].shape    
                else:
                    current_shape = denoised_blocks[x, y, z].shape                
                real_shape = np.array(current_shape) - 2 * overlap

                if x == 0:
                    real_shape[0] += overlap
                if y == 0:
                    real_shape[1] += overlap
                if z == 0:
                    real_shape[2] += overlap
                if x == x_size - 1:
                    real_shape[0] += overlap
                if y == y_size - 1:
                    real_shape[1] += overlap
                if z == z_size - 1:
                    real_shape[2] += overlap

                for i in range(begin_i, min(current_shape[0], final_size[0] - start_write_x)):                    
                    for j in range(begin_j, min(current_shape[1], final_size[1] - start_write_y)):                        
                        for k in range(begin_k, min(current_shape[2], final_size[2] - start_write_z)): 
                            if withChannels :
                                value = denoised_blocks[x, y, z][:, i, j, k]
                            else:
                                value = denoised_blocks[x, y, z][i, j, k]
                            coef = 1 
                            
                            if x > 0 and i < overlap:        
                                coef *= fenetre[i]
                            if y > 0 and j < overlap:        
                                coef *= fenetre[j]
                            if z > 0 and k < overlap:        
                                coef *= fenetre[k]

                            if x < x_size - 1:
                                if i >= overlap + blocks_sizes_x[x] or (x == 0 and i >= blocks_sizes_x[x]):        
                                    coef *= fenetre[current_shape[0] - 1 - i]
                            if y < y_size - 1:
                                if j >= overlap + blocks_sizes_y[y] or (y == 0 and j >= blocks_sizes_y[y]):
                                    coef *= fenetre[current_shape[1] - 1- j]
                            if z < z_size - 1:
                                if k >= overlap + blocks_sizes_z[z] or (z == 0 and k >= blocks_sizes_z[z]):
                                    coef *= fenetre[current_shape[2] - 1 - k]
                            if withChannels :
                                for channel in range(values[:, start_write_x + i, start_write_y + j, start_write_z + k].shape[0]):
                                    if values[channel, start_write_x + i, start_write_y + j, start_write_z + k] == None:
                                        values[channel, start_write_x + i, start_write_y + j, start_write_z + k] = []
                                    values[channel, start_write_x + i, start_write_y + j, start_write_z + k].append([value, coef])
                            else:
                                if values[start_write_x + i, start_write_y + j, start_write_z + k] == None:
                                    values[start_write_x + i, start_write_y + j, start_write_z + k] = []
                                values[start_write_x + i, start_write_y + j, start_write_z + k].append([value, coef])
                
                tmp =  denoised_blocks[x, y, z]
                denoised_blocks[x, y, z] = None
                del tmp
                gc.collect()


                start_write_z += max(blocks_sizes_z[z] - (z == 0) * overlap, 0)
                begin_k = max(0, overlap - blocks_sizes_z[z])

            start_write_y += max(blocks_sizes_y[y] - (y == 0) * overlap, 0)
            begin_j = max(0, overlap - blocks_sizes_y[y])

        start_write_x += max(blocks_sizes_x[x] - (x == 0) * overlap, 0)
        begin_i = max(0, overlap - blocks_sizes_x[x])
              
    for x in range(int(final_size[0])):
        for y in range(int(final_size[1])):
            for z in range(int(final_size[2])):                
                if withChannels :
                    for channel in range(values[:, x, y, z].shape[0]):                        
                        values[channel, x, y, z] = meanCustom(values[channel, x, y, z], methode = moyennage)
                else:
                    values[x, y, z] = meanCustom(values[x, y, z], methode = moyennage)

    return np.array(values, dtype = float)
