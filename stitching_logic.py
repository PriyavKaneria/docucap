import cv2
import numpy as np
import os
import glob
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# ==============================================================================
#  A NEW, GLOBALLY OPTIMIZED STITCHING IMPLEMENTATION
#  This version finds a central anchor image and aligns all others to it,
#  preventing error accumulation and producing a sharp result.
# ==============================================================================

class RobustStitcher:
    def __init__(self):
        self.finder = cv2.SIFT_create()

    def stitch(self, images):
        """
        Performs a robust, high-quality stitch using a central anchor and global alignment.
        """
        if len(images) < 2:
            return images[0] if images else None

        print("Starting robust global alignment stitching pipeline...")

        # 1. Feature Detection for all images
        print("Step 1: Detecting features...")
        keypoints, descriptors = [], []
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, des = self.finder.detectAndCompute(gray, None)
            keypoints.append(kp)
            descriptors.append(des)

         # 2. More Robust Feature Matching (FLANN + Lowe's Ratio Test)
        print("Step 2: Matching all pairs with robust filtering...")
        
        # FLANN parameters for SIFT
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        
        # Create the FLANN matcher
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        all_matches = {}
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                des1, des2 = descriptors[i], descriptors[j]
                
                # Ensure descriptors are valid and of type float32 for FLANN
                if (des1 is None or des2 is None or
                    des1.shape[0] < 2 or des2.shape[0] < 2):
                    continue

                # knnMatch expects float32 descriptors
                raw_matches = flann.knnMatch(des1.astype(np.float32), des2.astype(np.float32), k=2)

                # Apply Lowe's Ratio Test to filter out ambiguous matches
                good_matches = []
                for m, n in raw_matches:
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)

                if len(good_matches) > 20:
                    all_matches[(i, j)] = good_matches

        if not all_matches:
            print("Error: Not enough good matches found between any image pairs. Try better source images.")
            return None

        # 3. Select the best anchor image (the one with the most strong links)
        print("Step 3: Selecting anchor image...")
        inlier_counts = {}
        for i in range(len(images)):
            inlier_counts[i] = sum(1 for (im1, im2) in all_matches if im1 == i or im2 == i)
        
        anchor_idx = max(inlier_counts, key=inlier_counts.get)
        print(f"Selected image {anchor_idx} as the anchor.")
        
        # 4. Compute all homographies relative to the anchor image
        print("Step 4: Calculating all homographies relative to the anchor...")
        final_homographies = [np.eye(3) for _ in range(len(images))]

        for i in range(len(images)):
            if i == anchor_idx:
                continue

            # We must correctly identify which image was the query and which was the train.
            pair = (min(i, anchor_idx), max(i, anchor_idx))
            matches = all_matches.get(pair, [])
            if len(matches) < 4:
                continue

            # Keypoints for the current image `i` and the anchor image
            kp_i = keypoints[i]
            kp_anchor = keypoints[anchor_idx]
            
            # We need to map `kp_i` (src) to `kp_anchor` (dst).
            # We check which image was the query (first in pair) and which was train (second).
            if pair[0] == i and pair[1] == anchor_idx:
                # Case 1: i was query, anchor was train
                src_pts = np.float32([kp_i[m.queryIdx].pt for m in matches])
                dst_pts = np.float32([kp_anchor[m.trainIdx].pt for m in matches])
            elif pair[0] == anchor_idx and pair[1] == i:
                # Case 2: anchor was query, i was train. We must swap the indices.
                src_pts = np.float32([kp_i[m.trainIdx].pt for m in matches])
                dst_pts = np.float32([kp_anchor[m.queryIdx].pt for m in matches])
            else:
                # This case should not be reached, but as a safeguard:
                continue
            
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is not None:
                final_homographies[i] = H

        # 5. Iteratively warp and blend onto a final canvas
        print("Step 5: Warping all images once and blending...")
        
        # Calculate final canvas size
        h, w = images[anchor_idx].shape[:2]
        corners = np.float32([[0,0], [0,h], [w,h], [w,0]]).reshape(-1,1,2)
        all_corners = []
        for i in range(len(images)):
            H_inv = np.linalg.inv(final_homographies[i])
            warped_corners = cv2.perspectiveTransform(corners, H_inv)
            all_corners.append(warped_corners)
            
        all_corners = np.concatenate(all_corners, axis=0)
        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

        canvas_size = (x_max - x_min, y_max - y_min)
        H_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)

        # Initialize final panorama and weight map for blending
        panorama = np.zeros((canvas_size[1], canvas_size[0], 3), np.float32)
        weight_map = np.zeros((canvas_size[1], canvas_size[0]), np.float32)

        # This combined mask will track all non-black pixels
        combined_mask = np.zeros((canvas_size[1], canvas_size[0]), np.uint8)

        for i, img in enumerate(images):
            H_final = H_translation.dot(final_homographies[i])
            warped_img = cv2.warpPerspective(img, H_final, canvas_size)
            
            h_img, w_img = img.shape[:2]
            mask = np.ones((h_img, w_img), dtype=np.uint8) * 255
            warped_mask = cv2.warpPerspective(mask, H_final, canvas_size)
            
            feather_mask = cv2.distanceTransform(warped_mask, cv2.DIST_L2, 5).astype(np.float32)

            for c in range(3):
                panorama[:, :, c] += warped_img[:, :, c] * feather_mask
            weight_map += feather_mask
            
            # Add the warped mask to our combined mask
            combined_mask = cv2.bitwise_or(combined_mask, warped_mask)
        
        normalized_panorama = cv2.divide(panorama, cv2.merge([weight_map, weight_map, weight_map]) + 1e-7)
        normalized_panorama = normalized_panorama.astype(np.uint8)

        # 6. Find the tightest bounding box around the content and crop
        print("Step 6: Cropping black borders...")
        
        # Find contours of the non-black regions
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("Warning: No content found to crop. Returning uncropped image.")
            return normalized_panorama
            
        # Find the bounding box that encloses all contours
        x_min_crop, y_min_crop, w_crop, h_crop = cv2.boundingRect(np.concatenate(contours))
        
        # Crop the panorama to this bounding box
        final_image = normalized_panorama[y_min_crop:y_min_crop+h_crop, x_min_crop:x_min_crop+w_crop]
        
        print("Stitching complete.")
        return final_image
    

# ==============================================================================
#  YOUR EXISTING WORKFLOW FUNCTIONS - Now calling the new stitcher
# ==============================================================================

def stitch_calibration_images(esp_id):
    """
    Reads calibration images and uses the new robust, global alignment stitcher.
    """
    print(f"Starting ROBUST panorama stitching for ESP {esp_id}...")
    image_dir = f"data/calibration/esp{esp_id}/"
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg"))) 
    
    if len(image_paths) < 2:
        print(f"Not enough images to stitch for ESP {esp_id}.")
        return None

    images = [cv2.imread(p) for p in image_paths]
    images = [img for img in images if img is not None and img.shape[0] > 0 and img.shape[1] > 0]

    if not images:
        print("No valid images found.")
        return None
        
    stitcher = RobustStitcher()
    panorama = stitcher.stitch(images)
    
    if panorama is not None:
        stitched_path = f"data/stitched_panoramas/esp{esp_id}_panorama.jpg"
        cv2.imwrite(stitched_path, panorama)
        print(f"Robust panorama for ESP {esp_id} saved to {stitched_path}")
        return stitched_path
    else:
        print(f"Failed to create robust panorama for ESP {esp_id}")
        return None

def calculate_homography_and_stitch(panorama_paths):
    """
    Stitches the three main panoramas and saves simple homographies for live view.
    """
    print(f"Stitching final panoramas from: {panorama_paths}")
    images = [cv2.imread(p) for p in panorama_paths]
    if any(img is None for img in images):
        print("Error: Could not load all panoramic images.")
        return None, None

    # High-quality stitch for the final result
    stitcher = RobustStitcher()
    final_stitch = stitcher.stitch(images)
    final_image_path = None
    if final_stitch is not None:
        final_image_path = f"data/final_stitched/final_stitched_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(final_image_path, final_stitch)
        print(f"Final high-quality stitched image saved to {final_image_path}")
    else:
        print("Failed to create the final high-quality stitched image.")

    # Simple homography calculation for the live view's `.npy` file
    print("Calculating simple homographies for live stitching...")
    # (Re-using some older, faster helper functions for this specific task)
    keypoints, matches = detect_and_match_features_fast(images)
    homographies = compute_homographies_fast(keypoints, matches)
    homography_save_path = "data/final_homography.npy"
    np.save(homography_save_path, homographies)
    print(f"Simple homography matrices for live stitching saved to {homography_save_path}")

    return final_image_path, homography_save_path

def stitch_single_frames(frame_paths):
    homography_path = "data/final_homography.npy"
    if not os.path.exists(homography_path):
        print("ERROR: Homography file not found.")
        return None

    images = [cv2.imread(p) for p in frame_paths]
    if any(img is None for img in images):
        return None
        
    homographies = np.load(homography_path)
    
    # Simple iterative stitch for the fast live preview
    pano = images[0]
    for i in range(len(images) - 1):
        if i < len(homographies):
            # Assuming H maps image i+1 TO image i
            pano = stitch_pair_fast(pano, images[i+1], np.linalg.inv(homographies[i]))

    if pano is not None:
        stitched_path = f"data/final_stitched/capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(stitched_path, pano)
        return stitched_path
    return None

# Helper for fast live stitching (uses simple blending)
def stitch_pair_fast(img1, img2, H):
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]])
    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
    result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
    return result

# You will need these helper functions for calculate_homography_and_stitch
def detect_and_match_features_fast(images):
    orb = cv2.ORB_create()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    def detect(img): return orb.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(detect, images))
    keypoints = [res[0] for res in results]
    descriptors = [res[1] for res in results]
    matches = []
    if all(d is not None for d in descriptors):
        matches = [matcher.knnMatch(descriptors[i], descriptors[i+1], k=2) for i in range(len(descriptors)-1)]
    return keypoints, matches

def compute_homographies_fast(keypoints_list, matches_list):
    homographies = []
    for i, matches in enumerate(matches_list):
        if not matches:
            homographies.append(np.eye(3))
            continue
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good) < 4:
            homographies.append(np.eye(3))
            continue
        src_pts = np.float32([keypoints_list[i][m.queryIdx].pt for m in good])
        dst_pts = np.float32([keypoints_list[i+1][m.trainIdx].pt for m in good])
        H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        homographies.append(H if H is not None else np.eye(3))
    return homographies

if __name__ == "__main__":
    stitch_calibration_images(1)