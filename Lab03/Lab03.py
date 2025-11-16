# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import os

# NOTE: 
"""
Images that have problems--> parameter that corrects:
medium_risk_16.jpg --> fixed changing from momentu of inertia to score based on distance from center and size of cluster
medium_risk_5.jpg --> Ncluster=3 fix
melanoma_11.jpg --> Ncluster=3 improves, and implemented the filling of holes in the mole mask

in general, Ncluster = 3 or 4 is the main tradeoff to have good results in most images.
"""

# I add here the hyperparams so it easy to change them
# NOTE: if i put 4 clusters maybe the shadows around some moles will be put in a separate cluster
NCLUSTER = 4  # number of clusters for KMeans
EPSILON = 6  # epsilon for DBSCAN
M = 30       # min_samples for DBSCAN
MARGIN = 10  # margin for cropping
DELTA = 3    # delta for median filter
PENALTY = 100.0  # penalty for distance from center in cluster selection
# for single image
IMAGE = 'melanoma_14.jpg'  # specify the image name here


#%%
folderpath = './moles/'
files = os.listdir(folderpath) # list of file names in the folder /moles/
#%%

plotfig = True
np.set_printoptions(precision=2) # use only two decimal digits when printing numbers
plt.close('all') # close previously opened pictures

# ifile = files.index('low_risk_4.jpg')
# filein=files[ifile] # file to be analyzed (low_risk, medium_risk or melanoma)
# print(filein)

def main(folderpath, filein , NCLUSTER, EPSILON, M, MARGIN, DELTA, PENALTY):
    print(f"Processing: {filein}")

    # check if file exists 
    try:
        im_or = mpimg.imread(os.path.join(folderpath, filein))
    except FileNotFoundError:
        print(f"File {filein} not found. Change dir before running the code. (cd Lab03)")
        return

    # Image Processing (Grayscale)
    # im_or is Ndarray N1 x N2 x 3 unint8 (integers from 0 to 255)
    # gray_image is an Ndaaray N1 x N2 unint8 (integers from 0 to 255)
    N1, N2, N3 = im_or.shape # NOTE: N3 is 3, the number of elementary colors, i.e. red, green ,blue
    gray_image = np.mean(im_or, axis=2).astype(np.uint8)
    N1, N2= gray_image.shape

    #%% get a simplified image with only Ncluster colors
    # K-Means Quantization
    # number of clusters/quantized colors we want to have in the simpified image:
    Ncluster = NCLUSTER

    # instantiate the object K-means:
    kmeans = KMeans(n_clusters=Ncluster, random_state=0, n_init=10)

    # run K-means on the colors of the gray image (i.e. on the uint8 values):
    im_1D = gray_image.reshape((N1*N2, 1))
    kmeans.fit(im_1D)


    # get the centroids (i.e. the 3 gray colors). Note that the centroids
    # take real values, we must convert these values to uint8
    # to properly see the quantized image
    kmeans_centroids = kmeans.cluster_centers_.astype('uint8')
    im_1D_quant = im_1D.copy() # copy im_1D into im_1D_quant and get the quantized image

    # number of kmeans_centroids might differ from the orginal Ncluster
    # for this reason, len(kmeans_centroids). Maybe less than Ncluster
    for kc in range(len(kmeans_centroids)):
        # substitute in im_1D_quant all pixels assigned to cluster kc with the centroid value
        im_1D_quant[(kmeans.labels_ == kc), :] = kmeans_centroids[kc, :]
    im_quant = im_1D_quant.reshape((N1, N2))

    
    # Identify Darkest Cluster
    #%% Preliminary steps to find the contour after the clustering
    # 1: find the darkest color found by k-means, since the darkest color
    # corresponds to the mole:
    i_col = kmeans_centroids.argmin() # darkest color --> minimum grayscale value

    # 2: define the 2D-array im_clust where in position i,j you have the index of
    # the cluster pixel i,j belongs to 
    im_clust = kmeans.labels_.reshape(N1, N2)

    # 3: find the positions i,j where im_clust is equal to i_col (cluster with the darkest color)
    # the 2D Ndarray mole_pos stores the coordinates i,j only of the pixels
    # in cluster i_col
    mole_pos = np.argwhere(im_clust == i_col)  # Ndarray with two columns, storing the index [i,j] of the dark pixels 

    
    # DBSCAN Segmentation
    #%% Find the likely position of the mole using DBSCAN
    epsilon = EPSILON
    M = M

    # fit DBSCAN on the positions of the dark pixels (closeby dark pixels will belong to the same cluster)
    clusters = DBSCAN(eps=epsilon, min_samples=M, metric='euclidean').fit(mole_pos)
    id_clusters, count_id_clusters = np.unique(clusters.labels_, return_counts=True) # count the number of obtained clusters (i.e. groups of closeby dark pixels)

    print('Number of points in each cluster found by DBSCAN: ',count_id_clusters)
    print('Indexes of the found clusters: ',id_clusters)

    # NOTE: the implementation of the Momentum of Inertia caused the wrong selection in some cases like medium_risk_16.jpg
    # Cluster Selection (Momentum of Inertia / SSE)
    # use this method to choose the cluster that most likely corresponds to the mole
    # best_cluster_id = -1
    # min_inertia = float('inf')

    # for label in id_clusters:
    #     if label == -1: continue 
        
    #     cluster_pixels = mole_pos[clusters.labels_ == label]
        
    #     # Filter small noise < 1000 pixels
    #     if len(cluster_pixels) < 1000:
    #         continue
            
    #     # Calculate Centroid
    #     centroid = np.mean(cluster_pixels, axis=0)
        
    #     # Calculate Inertia (Sum of Squared Errors)
    #     diff = cluster_pixels - centroid
    #     dist_sq = np.sum(diff**2, axis=1)
    #     inertia = np.sum(dist_sq)
        
    #     if inertia < min_inertia:
    #         min_inertia = inertia
    #         best_cluster_id = label

    # i_mole = best_cluster_id

    # Cluster Selection
    best_cluster_id = -1
    best_score = -float('inf')
    
    # method based on size and distance from center (replaced inertia)
    # Image center (where the mole is)
    center_img = np.array([N1/2, N2/2]) 
    
    # the further from the center, the higher the penalty --> avoid taking shadows
    distance_penalty = PENALTY

    for label in id_clusters:
        if label == -1: continue 
        
        cluster_pixels = mole_pos[clusters.labels_ == label]
        
        # exclude noise or small moles like in image medium_risk_16.jpg
        n_pixels = len(cluster_pixels)
        if n_pixels < 1000: continue
            
        # 1. Calculate the centroid of the cluster
        centroid = np.mean(cluster_pixels, axis=0)
        
        # 2. Calculate distance from image center
        dist = np.linalg.norm(centroid - center_img)
        
        # 3. Calculate SCORE
        # The larger (+ n_pixels), the higher the score.
        # The further (- dist * penalty), the lower the score.
        score = n_pixels - (dist * distance_penalty)
        
        # print(f"Cluster {label}: Size={n_pixels}, Dist={dist:.1f}, Score={score:.1f}")

        if score > best_score:
            best_score = score
            best_cluster_id = label

    i_mole = best_cluster_id
    
    if i_mole == -1: 
        print(f"No valid mole cluster found for {filein}")
        return

    # Extract the mole pixels
    true_mole_pos = mole_pos[clusters.labels_ == i_mole] # pixel indexes [i,j] in the cluster that corresponds to the mole
    
    # Create visualization images for this step
    im_only_mole_gray = np.zeros((N1, N2), dtype=float) + 255 # White background
    x = true_mole_pos[:, 0]
    y = true_mole_pos[:, 1]
    im_only_mole_gray[x, y] = gray_image[x, y]
    
    im_mole_pos = np.ones((N1, N2), dtype='uint8') * 255 # * 255 is used to create a white color
    im_mole_pos[x, y] = 0 # black where the mole is present

    ###################################################
    im_mole_pos = np.ones((N1, N2), dtype='uint8') * 255
    im_mole_pos[x, y] = 0 

    # Fill Holes in the Mole Mask
    # Create copies for horizontal and vertical filling
    fill_row = im_mole_pos.copy()
    fill_col = im_mole_pos.copy()

    # Horizontal Fill (Scan Rows)
    # Find the first and last black pixel in every row and fill between them
    for r in range(N1):
        idxs = np.where(im_mole_pos[r, :] == 0)[0] # Find black pixels
        if len(idxs) > 1:
            fill_row[r, idxs[0]:idxs[-1]] = 0 # Set interval to Black (0)

    # Vertical Fill (Scan Columns)
    # Find the first and last black pixel in every column and fill between them
    for c in range(N2):
        idxs = np.where(im_mole_pos[:, c] == 0)[0] 
        if len(idxs) > 1:
            fill_col[idxs[0]:idxs[-1], c] = 0

    # Intersection
    # A pixel is a hole ONLY if it is filled in BOTH directions.
    # - If Row=0 (Mole) and Col=0 (Mole) -> Max=0 (Mole) --> Hole filled
    # - If Row=0 (Mole) but Col=255 (Skin) -> Max=255 (Skin) --> Unchanged
    im_mole_pos = np.maximum(fill_row, fill_col)
    
    # Cropping
    #%% Find the cropped original image
    margin = MARGIN # pixels around the mole (used for smoothing the image)
    min_x = max(0, np.min(x) - margin)
    max_x = min(N1, np.max(x) + margin)
    min_y = max(0, np.min(y) - margin)
    max_y = min(N2, np.max(y) + margin)
    
    im_cropped_gray_red = im_only_mole_gray[min_x:max_x, min_y:max_y]
    im_cropped_col = im_or[min_x:max_x, min_y:max_y, :]
    im_cropped_mole_pos = im_mole_pos[min_x:max_x, min_y:max_y]

    
    # Smoothing (Median Filter)
    #%% Smooth the image
    delta = DELTA
    N1_c, N2_c = im_cropped_mole_pos.shape
    im_cropped_mole_pos_filt = np.ones_like(im_cropped_mole_pos) * 255 # white image
    
    for kr in range(delta, N1_c - delta):
        for kc in range(delta, N2_c - delta):
            sub = im_cropped_mole_pos[kr-delta:kr+delta+1, kc-delta:kc+delta+1]
            im_cropped_mole_pos_filt[kr, kc] = np.median(sub)

    
    #%% Apply Sobel filters
    kern1 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) # Sobel filter Hx
    kern2 = kern1.T # Sobel filter Hy

    image_for_filtering = im_cropped_mole_pos_filt.astype(float)
    Gx = np.zeros_like(image_for_filtering)
    Gy = np.zeros_like(image_for_filtering)
    
    rows, cols = image_for_filtering.shape
    
    # Manual Convolution Loop
    for kr in range(1, rows - 1):
        for kc in range(1, cols - 1):
            sub_img = image_for_filtering[kr-1:kr+2, kc-1:kc+2]
            Gx[kr, kc] = np.sum(sub_img * kern1)
            Gy[kr, kc] = np.sum(sub_img * kern2)

    G = np.sqrt(Gx**2 + Gy**2)
    
    threshold = np.max(G) * 0.5 
    border = np.zeros_like(G)
    border[G > threshold] = 1 

    
    # 9. Visualization (All in one Figure)
    
    if plotfig:
        # Create a grid of 2 rows x 3 columns
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs = axs.ravel() # Flatten the grid for easy indexing
        
        # 1. Original Image
        axs[0].imshow(im_or)
        axs[0].set_title(f'1. Original: {filein}')
        
        # 2. Quantized Image
        axs[1].imshow(im_quant, cmap='gray')
        axs[1].set_title('2. Quantized (K-Means)')
        
        # 3. Segmented Mole (DBSCAN Result)
        axs[2].imshow(im_only_mole_gray, cmap='gray')
        axs[2].set_title('3. DBSCAN Selected')
        
        # 4. Cropped Mask (Before Smoothing)
        axs[3].imshow(im_cropped_mole_pos, cmap='gray')
        axs[3].set_title('4. Cropped Mask')
        
        # 5. Smoothed Mask (Median Filter)
        axs[4].imshow(im_cropped_mole_pos_filt, cmap='gray')
        axs[4].set_title('5. Smoothed Mask')
        
        # 6. Final Result (Border Overlay)
        axs[5].imshow(im_cropped_col, interpolation='none')
        # Mask zeros in border so they are transparent
        border_masked = np.ma.masked_where(border == 0, border)
        axs[5].imshow(border_masked, cmap='spring', interpolation='none', alpha=1.0, vmin=0, vmax=1)
        axs[5].set_title('6. Final Border (Sobel)')
        
        # Remove axis ticks for cleaner view
        for ax in axs:
            ax.axis('off')
            
        plt.tight_layout()
        plt.show()

    return None


# Run ALL images in the folder
for ifile_name in files:
    if ifile_name.lower().endswith(('.jpg', '.jpeg')):
        main(folderpath, ifile_name, NCLUSTER, EPSILON, M, MARGIN, DELTA, PENALTY)

# Run specified image
# single_file_name = IMAGE # TODO change with the specified name
# main(folderpath, single_file_name, NCLUSTER, EPSILON, M, MARGIN, DELTA, PENALTY)