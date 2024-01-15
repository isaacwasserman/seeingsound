from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d, convolve1d
import time
import lap
from pathlib import Path
import numpy as np
import cv2

''' Plots any given number of images'''
def plot_grid(*images, figsize=6, fignumber="Filter", titles=None, occurences=False):
    num_plots = len(images)
    
    plt.close(fignumber)
    fig = plt.figure(figsize=(figsize*int(min(num_plots, 5)), figsize*int(max(num_plots//5, 1))), num=fignumber)

    for i, grid in enumerate(images):
        
        size = grid.shape
        
        if size[-1] == 1:
            if occurences:
                cmap=None
            else:
                cmap="gray"
        else:
            cmap=None
        
        if len(size) == 3:
            ax = fig.add_subplot(((num_plots - 1) // 5) + 1, min(int(num_plots % 5) + (int(num_plots // 5) * 5), 5), i+1)
            img = grid.reshape(*size)
            ax.imshow(np.squeeze(img), cmap=cmap, vmin=0)
            ax.set_xticks([])
            ax.set_yticks([])
             
        if titles is not None:
            ax.set_title(titles[i], fontsize=figsize*3)
    
    plt.show()

''' Generates feature vectors from small thumbnail images '''
def create_vectors_from_thumbnails(image_files, thumb_size=4):
    
    n_images = len(image_files)   
    feature_vectors = np.zeros((n_images, thumb_size*thumb_size*3))

    try:
        for i in range(n_images):
            file = image_files[i]
            im = Image.open(file)
            im = im.convert("RGB")
            im = im.resize((thumb_size, thumb_size), Image.LANCZOS)

            # create dct for each channel
            dct = np.zeros((thumb_size, thumb_size, 3))
            for c in range(3):
                dct[:, :, c] = cv2.dct(np.array(im)[:, :, c]/255)

            pixels = np.array(im) / 255
            pixels = pixels.reshape((thumb_size * thumb_size * 3))
        
            feature_vectors[i] = pixels
            feature_vectors[i] = dct.reshape((thumb_size * thumb_size * 3))
    except Exception as e:
        pass
            
    return feature_vectors

def list_all_image_paths(images_path):
    return sorted([str(f) for f in Path(images_path).rglob('*.*') if (("jpeg" in str(f).lower()) | ("jpg" in str(f).lower()) | ("png" in str(f).lower())) & ("._" not in str(f).lower()) & (".txt" not in str(f).lower())])


def create_collage(sorted_1d_filepaths, pixels_per_image=32, grid_shape=None):
    reshaped_files = np.array(sorted_1d_filepaths).reshape((grid_shape[0], grid_shape[1]))

    X = np.zeros((pixels_per_image*grid_shape[0], pixels_per_image*grid_shape[1], 3))

    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            try:
                file = reshaped_files[i, j]
                im = Image.open(file)
                im = im.convert("RGB")
                im = im.resize((pixels_per_image, pixels_per_image), Image.LANCZOS)
                pixels = np.array(im) / 255

                X[i*pixels_per_image:(i+1)*pixels_per_image, j*pixels_per_image:(j+1)*pixels_per_image] = pixels
            except Exception as e:
                print(e)
    return X

''' Calculates the squared L2 (eucldean) distance using numpy. '''
def squared_l2_distance(q, p):
    
    ps = np.sum(p*p, axis=-1, keepdims=True)    
    qs = np.sum(q*q, axis=-1, keepdims=True)
    distance = ps - 2*np.matmul(p, q.T) + qs.T
    return np.clip(distance, 0, np.inf)
''' Applies a low pass filter to the current map'''
def low_pass_filter(map_image, filter_size_x, filter_size_y, wrap=False):
    
    mode = "wrap" if wrap else "reflect" # nearest
    
    # print(f"Filter size: {filter_size_x}x{filter_size_y}")
    # print(map_image.shape)
    im2 = uniform_filter1d(map_image, filter_size_y, axis=0, mode=mode)  
    im2 = uniform_filter1d(im2, filter_size_x, axis=1, mode=mode)  
    return im2
''' Utility function that takes a position and returns 
a desired number of positions in the given radius'''
def get_positions_in_radius(pos, indices, r, nc, wrap):
    if wrap:
        return get_positions_in_radius_wrapped(pos, indices, r, nc)
    else:
        return get_positions_in_radius_non_wrapped(pos, indices, r, nc)
''' Utility function that takes a position and returns 
a desired number of positions in the given radius'''
def get_positions_in_radius_non_wrapped(pos, indices, r, nc):
    
    H, W = indices.shape
    
    x = pos % W 
    y = int(pos/W)
    
    ys = y-r
    ye = y+r+1
    xs = x-r
    xe = x+r+1
    
    # move position so the full radius is inside the images bounds
    if ys < 0:
        ys = 0
        ye = min(2*r + 1, H)
        
    if ye > H:
        ye = H
        ys = max(H - 2*r - 1, 0)
        
    if xs < 0:
        xs = 0
        xe = min(2*r + 1, W)
        
    if xe > W:
        xe = W
        xs = max(W - 2*r - 1, 0)
    
    # concatenate the chosen position to a 1D array
    positions = np.concatenate(indices[ys:ye, xs:xe])
    
    if nc is None:
        return positions
    
    chosen_positions = np.random.choice(positions, min(nc, len(positions)), replace=False)
    
    return chosen_positions
''' Utility function that takes a position and returns 
a desired number of positions in the given radius'''
def get_positions_in_radius_wrapped(pos, extended_grid, r, nc):
    
    H, W = extended_grid.shape
    
    # extended grid shape is H*2, W*2
    H, W = int(H/2), int(W/2)    
    x = pos % W 
    y = int(pos/W)
    
    ys = (y-r + H) % H     
    ye = ys + 2*r + 1 
    xs = (x-r + W) % W 
    xe = xs + 2*r + 1 
    
    # concatenate the chosen position to a 1D array
    positions = np.concatenate(extended_grid[ys:ye, xs:xe])
    
    if nc is None:
        return positions
    
    chosen_positions = np.random.choice(positions, min(nc, len(positions)), replace=False)
    
    return chosen_positions
def sort_with_las(X, filepaths, radius_factor = 0.9, wrap = False, grid_shape=None):
    
    # for reproducible sortings
    np.random.seed(7)
    filepaths = np.array(filepaths)
    
    # n_images_per_site = int(np.sqrt(len(X)))
    
    # X = X.reshape((n_images_per_site, n_images_per_site, -1))
    
    # setup of required variables
    N = np.prod(X.shape[:-1])
    X = X.reshape((grid_shape[0], grid_shape[1], -1))
    filepaths = np.array(filepaths)
   
    grid_shape = X.shape[:-1]
    H, W = grid_shape
    start_time = time.time()
    # assign input vectors to random positions on the grid
    grid = np.random.permutation(X.reshape((N, -1))).reshape((X.shape)).astype(float)
    # reshape 2D grid to 1D
    flat_X = X.reshape((N, -1))
    
    radius_f = max(H, W)/2 - 1 
    
    while True:
        print(".", end="")
        # compute filtersize that is not larger than any side of the grid
        radius = int(np.round(radius_f))
        
        filter_size_x = min(W-1, int(2*radius + 1))
        filter_size_y = min(H-1, int(2*radius + 1))
        #print (f"radius {radius_f:.2f} Filter size: {filter_size_x}")
        
        # Filter the map vectors using the actual filter radius
        grid = low_pass_filter(grid, filter_size_x, filter_size_y, wrap=wrap)
        flat_grid = grid.reshape((N, -1))
              
        # calc C
        pixels = flat_X
        grid_vecs = flat_grid
        C = squared_l2_distance(pixels, grid_vecs)
        # quantization of distances speeds up assigment solver
        C = (C / C.max() * 2048).astype(int)
        
        # get indices of best assignements 
        _, best_perm_indices, _= lap.lapjv(C)
        
        #Assign the input vectors to their new map positions in 1D
        flat_X = pixels[best_perm_indices]
        filepaths = filepaths[best_perm_indices]
        #print(filepaths)
        # prepare variables for next iteration
        grid = flat_X.reshape(X.shape)
        
        radius_f *= radius_factor
        if radius_f < 1:
            break
                
    print(f"\nSorted with LAS in {time.time() - start_time:.3f} seconds") 
    
    # return sorted grid
    return grid, filepaths
def sort_with_flas(X, filepaths, nc, radius_factor=0.9, wrap=False, return_time=False, grid_shape=None):
    
    np.random.seed(7)   # for reproducible sortings
    
    # setup of required variables
    N = np.prod(X.shape[:-1])
    
    if grid_shape is None:
        grid_shape = (int(np.sqrt(len(X))), int(np.sqrt(len(X))))

    # n_images_per_site = int(np.sqrt(len(X)))
    X = X.reshape((grid_shape[0], grid_shape[1], -1))
    filepaths = np.array(filepaths)
   
    grid_shape = X.shape[:-1]
    H, W = grid_shape
    
    start_time = time.time()
    
    # assign input vectors to random positions on the grid
    grid = np.random.permutation(X.reshape((N, -1))).reshape((X.shape)).astype(float)
    
    # reshape 2D grid to 1D
    flat_X = X.reshape((N, -1))
    
    # create indices array 
    indices = np.arange(N).reshape(grid_shape)
    
    if wrap:
        # create a extended grid of size (H*2, W*2)
        indices = np.concatenate((indices, indices), axis=1 )
        indices = np.concatenate((indices, indices), axis=0 )
    
    radius_f = max(H, W)/2 - 1 # initial radius
        
    while True:
        # compute filtersize that is smaller than any side of the grid
        radius = int(radius_f)
        filter_size_x = max(1, min(W-1, int(2*radius + 1)))
        filter_size_y = max(1, min(H-1, int(2*radius + 1)))
        
        # Filter the map vectors using the actual filter radius
        grid = low_pass_filter(grid, filter_size_x, filter_size_y, wrap=wrap)
        flat_grid = grid.reshape((N, -1))
        
        n_iters = 2 * int(N / nc) + 1
        max_swap_radius = int(round(max(radius, (np.sqrt(nc)-1)/2)))
            
        for i in range(n_iters):
            
            # find random swap candicates in radius of a random position
            random_pos = np.random.choice(N, size=1)
            positions = get_positions_in_radius(random_pos[0], indices, max_swap_radius, nc, wrap=wrap)
            
            # calc C
            pixels = flat_X[positions]
            grid_vecs = flat_grid[positions]
            C = squared_l2_distance(pixels, grid_vecs)
            
            # quantization of distances speeds up assingment solver
            C = (C / C.max() * 2048).astype(int)
            
            # get indices of best assignments 
            _, best_perm_indices, _= lap.lapjv(C)
            
            # assign the input vectors to their new map positions
            flat_X[positions] = pixels[best_perm_indices]
            filepaths[positions] = filepaths[positions][best_perm_indices]
        
         # prepare variables for next iteration
        grid = flat_X.reshape(X.shape)
        
        radius_f *= radius_factor
        # break condition
        if radius_f < 1:
            break
               
    duration = time.time() - start_time
    
    if return_time:
        return grid, filepaths, duration
    
    print(f"Sorted with FLAS in {duration:.3f} seconds") 
    return grid, filepaths

def compute_spatial_distances_for_grid(grid_shape, wrap):
    if wrap:
        return compute_spatial_distances_for_grid_wrapped(grid_shape)
    else:
        return compute_spatial_distances_for_grid_non_wrapped(grid_shape)
def compute_spatial_distances_for_grid_wrapped(grid_shape):
    
    n_x = grid_shape[0]
    n_y = grid_shape[1]

    wrap1 = [[0,   0], [0,   0], [0,     0], [0, n_y], [0,   n_y], [n_x, 0], [n_x,   0], [n_x, n_y]]
    wrap2 = [[0, n_y], [n_x, 0], [n_x, n_y], [0,   0], [n_x,   0], [  0, 0], [  0, n_y], [  0,   0]]
    
    # create 2D position matrix with tuples, i.e. [(0,0), (0,1)...(H-1, W-1)]
    a, b = np.indices(grid_shape)
    mat = np.concatenate([np.expand_dims(a, -1), np.expand_dims(b, -1)], axis=-1)
    mat_flat = mat.reshape((-1, 2))
    
    # use this 2D matrix to calculate spatial distances between on positions on the grid
    d = squared_l2_distance(mat_flat, mat_flat)
    for i in range(8):
        # look for smaller distances with wrapped coordinates
        d_i = squared_l2_distance(mat_flat + wrap1[i], mat_flat + wrap2[i])
        d = np.minimum(d, d_i)
        
    return d
def compute_spatial_distances_for_grid_non_wrapped(grid_shape):
    
    # create 2D position matrix with tuples, i.e. [(0,0), (0,1)...(H-1, W-1)]
    a, b = np.indices(grid_shape)
    mat = np.concatenate([np.expand_dims(a, -1), np.expand_dims(b, -1)], axis=-1)
    mat_flat = mat.reshape((-1, 2))
    
    # use this 2D matrix to calculate spatial distances between on positions on the grid
    d = squared_l2_distance(mat_flat, mat_flat)
    return d
''' sorts a matrix so that row values are sorted by the 
    spatial distance and in case they are equal, by the HD distance '''
def sort_hddists_by_2d_dists(hd_dists, ld_dists):

    max_hd_dist = np.max(hd_dists) * 1.0001

    ld_hd_dists = hd_dists/max_hd_dist + ld_dists # add normed HD dists (0 .. 0.9999) to the 2D int dists
    ld_hd_dists = np.sort(ld_hd_dists)  # then a normal sorting of the rows can be used

    sorted_HD_D = np.fmod(ld_hd_dists, 1) * max_hd_dist
    
    return sorted_HD_D
''' computes the Distance Preservation Gain delta DP_k(S) '''
def get_distance_preservation_gain(sorted_d_mat, d_mean):
    
    # range of numbers [1, K], with K = N-1
    nums = np.arange(1, len(sorted_d_mat))
    
    # compute cumulative sum of neighbor distance values for all rows, shape = (N, K)
    cumsum = np.cumsum(sorted_d_mat[:, 1:], axis=1)
    
    # compute average of neighbor distance values for all rows, shape = (N, K)
    d_k = (cumsum / nums)
    
    # compute average of all rows for each k, shape = (K, )
    d_k = d_k.mean(axis=0)
    
    # compute Distance Preservation Gain and set negative values to 0, shape = (K, )
    d_k = np.clip((d_mean - d_k) / d_mean, 0, np.inf)
   
    return d_k 
''' computes the Distance Preservation Quality DPQ_p(S)'''
def distance_preservation_quality(sorted_X, p=2, wrap=False):
    
    # setup of required variables
    grid_shape = sorted_X.shape[:-1]
    N = np.prod(grid_shape)
    H, W = grid_shape
    flat_X = sorted_X.reshape((N, -1))
    
    # compute matrix of euclidean distances in the high dimensional space
    dists_HD = np.sqrt(squared_l2_distance(flat_X, flat_X)) 
    
    # sort HD distance matrix rows in acsending order (first value is always 0 zero now)
    sorted_D = np.sort(dists_HD, axis=1)
    
    # compute the expected value of the HD distance matrix
    mean_D = sorted_D[:, 1:].mean()
    
    # compute spatial distance matrix for each position on the 2D grid
    dists_spatial = compute_spatial_distances_for_grid(grid_shape, wrap)
    
    # sort rows of HD distances by the values of spatial distances
    sorted_HD_by_2D = sort_hddists_by_2d_dists(dists_HD, dists_spatial)
    
    # get delta DP_k values
    delta_DP_k_2D = get_distance_preservation_gain(sorted_HD_by_2D, mean_D)
    delta_DP_k_HD = get_distance_preservation_gain(sorted_D, mean_D)
     
    # compute p norm of DP_k values
    normed_delta_D_2D_k = np.linalg.norm(delta_DP_k_2D, ord=p)
    normed_delta_D_HD_k = np.linalg.norm(delta_DP_k_HD, ord=p)
    
    # DPQ(s) is the ratio between the two normed DP_k values
    DPQ = normed_delta_D_2D_k/normed_delta_D_HD_k
   
    return DPQ

# for rf in [0.3, 0.5, 0.7, 0.95]:
#     for nc in [9, 16, 25, 49, 81]:
#         print(".", end="")
#         # sort random image with current FLAS parameters
#         sorted_X, sorted_image_paths, duration = sort_with_flas(vectors.copy(), image_files, radius_factor=rf, nc=nc, return_time=True)
        
#         collage = create_collage(sorted_image_paths)
#         # store sorted image
#         images.append(collage)
        
#         # get DPQ_p(S)
#         p = 16
#         dpq = distance_preservation_quality(sorted_X, p=p)
#         print(". ", end="")
        
#         # create string with parameters, duration and DPQ
#         titles.append(f"nc: {nc}, radius factor: {rf}\n time: {duration:0.3f}s, DPQ_{p}: {dpq:0.3f}")

# for rf in [0.3, 0.5, 0.7, 0.95]:
#     for nc in [9, 16, 25, 49, 81]:
#         print(".", end="")
#         # sort random image with current FLAS parameters
#         sorted_X, sorted_image_paths, duration = sort_with_flas(vectors.copy(), image_files, radius_factor=rf, nc=nc, return_time=True, wrap=True)
        
#         collage = create_collage(sorted_image_paths)
#         # store sorted image
#         images.append(collage)
        
#         # get DPQ(S)
#         p = 16
#         dpq = distance_preservation_quality(sorted_X, p=16)
#         print(". ", end="")
        
#         # create string with parameters, duration and DPQ
#         titles.append(f"nc: {nc}, radius factor: {rf}\n time: {duration:0.3f}s, DPQ_{p}: {dpq:0.3f}")

def is_prime(n):
    for i in range(2, int(np.sqrt(n))+1):
        if n % i == 0:
            return False
    return True

def closest_factor_pair(n):
    distance = n - 1
    best = (1, n)
    for i in range(1, int(np.sqrt(n))+1):
        if n % i == 0:
            if abs(i - n/i) < distance:
                best = (i, n/i)
                distance = abs(i - n/i)
    return best

def assign_positions_to_images(image_dir, grid_shape=None, feature_vector_size=4, pad=True, max_images=None):
    set_folder = image_dir
    image_files = list_all_image_paths(set_folder)
    image_files = [path for path in image_files if " " not in path]
    # If prime number of images, repeat last image until not prime
    num_added = 0
    while min(closest_factor_pair(len(image_files))) < 4:
        image_files += [image_files[-1]]
        num_added += 1
    print(f"Added {num_added} images to make grid shape possible")

    print(f"number of paths: {len(image_files)}")
    print(f"number of unique paths: {len(set(image_files))}")

    if max_images is not None and len(image_files) > max_images:
        # randomly select max_images from image_files
        image_files = np.random.choice(image_files, max_images, replace=False)

    if grid_shape is not None:
        if grid_shape[0] == -1 and grid_shape[1] == -1:
            grid_shape = None
        elif grid_shape[0] == -1:
            grid_shape = (len(image_files) // grid_shape[1] + 1, grid_shape[1])
        elif grid_shape[1] == -1:
            grid_shape = (grid_shape[0], len(image_files) // grid_shape[0] + 1)
    elif grid_shape is None:
        root = np.sqrt(len(image_files))
        best = 1
        best_distance = abs(root - best)
        for i in range(1, int(root)+1):
            if len(image_files) % i == 0 and abs(root - i) < best_distance:
                best = i
                best_distance = abs(root - i)
        grid_shape = (best, len(image_files) // best)

    if grid_shape[0] * grid_shape[1] < len(image_files):
        print("Grid shape is too small for the number of images")
        return
    elif grid_shape[0] * grid_shape[1] > len(image_files):
        if not pad:
            print("Grid shape is too large for the number of images")
            return
        else:
            image_files += ["blank.png"] * (grid_shape[0] * grid_shape[1] - len(image_files))
            blank = np.ones((128, 128, 3)) * 255
            plt.imsave("blank.png", blank.astype(np.uint8))

    print(f"Grid shape: {grid_shape}")
    print(f"Number of images: {len(image_files)}")

    vectors = create_vectors_from_thumbnails(image_files, thumb_size=feature_vector_size)
    _, sorted_files = sort_with_las(vectors.copy(), image_files, radius_factor=0.7, wrap=True, grid_shape=grid_shape)

    sorted_ids = [f.split("/")[-1].split(".")[0] for f in sorted_files]
    print(f"number of ids: {len(sorted_ids)}")
    print(f"number of unique ids: {len(set(sorted_ids))}")
          
    gridded_ids = np.array(sorted_ids).reshape(grid_shape)
    coords_by_id = {}
    ids_by_coords = {}
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            id = gridded_ids[i, j]
            coords_by_id[id] = (i, j)
            ids_by_coords[(i, j)] = id
    # print(gridded_ids)
    # save gridded ids to csv
    np.savetxt("gridded_ids.csv", gridded_ids, delimiter=",", fmt="%s")


    collage = create_collage(sorted_files, grid_shape=grid_shape)
    # X = create_collage(image_files, grid_shape=grid_shape)
    plt.imsave("FLAS.png", collage)

    # plot_grid(collage, figsize=6, titles=["FLAS sorted Arrangement"])

# assign_positions_to_images("polished/flattened", feature_vector_size=8, max_images=1024)