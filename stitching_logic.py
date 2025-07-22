import cv2
import numpy as np
import os
import glob
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

def is_frame_sane(frame, min_std_dev=12.0):
    """
    Checks if a frame is likely valid and not a solid color "warm-up" frame.
    A valid image should have a reasonable amount of detail, which corresponds
    to a higher standard deviation of pixel values.
    """
    if frame is None:
        return False
    # Check for minimal dimensions
    if frame.shape[0] < 100 or frame.shape[1] < 100:
        return False
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Calculate the standard deviation of pixel intensities
    std_dev = np.std(gray)
    if std_dev < min_std_dev:
        print(f"Skipping insane frame with low detail (std dev: {std_dev:.2f})")
        return False
    return True

# ==============================================================================
#  ACCURATE, GLOBALLY OPTIMIZED STITCHER (for offline processing)
# ==============================================================================
class RobustStitcher:
    def __init__(self, match_confidence=0.7, ransac_thresh=3.0):
        self.finder = cv2.SIFT_create()
        self.match_confidence = match_confidence
        self.ransac_thresh = ransac_thresh

    def stitch(self, images):
        if len(images) < 2: return images[0] if images else None
        print("Starting robust global alignment stitching...")

        keypoints, descriptors = [], []
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, des = self.finder.detectAndCompute(gray, None)
            keypoints.append(kp)
            descriptors.append(des)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        all_matches = {}
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                des1, des2 = descriptors[i], descriptors[j]
                if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2: continue
                raw_matches = flann.knnMatch(des1, des2, k=2)
                good_matches = [m for m, n in raw_matches if len(raw_matches) > 1 and m.distance < self.match_confidence * n.distance]
                if len(good_matches) > 20: all_matches[(i, j)] = good_matches
        
        if not all_matches: return None
        
        inlier_counts = {i: sum(1 for (im1, im2) in all_matches if im1 == i or im2 == i) for i in range(len(images))}
        anchor_idx = max(inlier_counts, key=inlier_counts.get)
        
        final_homographies = [np.eye(3) for _ in images]
        for i in range(len(images)):
            if i == anchor_idx: continue
            pair = (min(i, anchor_idx), max(i, anchor_idx))
            matches = all_matches.get(pair, [])
            if len(matches) < 4: continue
            kp_i, kp_anchor = keypoints[i], keypoints[anchor_idx]
            
            if pair[0] == i:
                src_pts = np.float32([kp_i[m.queryIdx].pt for m in matches])
                dst_pts = np.float32([kp_anchor[m.trainIdx].pt for m in matches])
            else:
                src_pts = np.float32([kp_i[m.trainIdx].pt for m in matches])
                dst_pts = np.float32([kp_anchor[m.queryIdx].pt for m in matches])
            
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_thresh)
            if H is not None: final_homographies[i] = H

        h, w = images[anchor_idx].shape[:2]
        corners = np.float32([[0,0], [0,h], [w,h], [w,0]]).reshape(-1,1,2)
        all_corners = [cv2.perspectiveTransform(corners, np.linalg.inv(H)) for H in final_homographies if H is not None]
        all_corners = np.concatenate(all_corners, axis=0)
        
        x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)
        T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

        canvas_size = (x_max - x_min, y_max - y_min)
        panorama = cv2.warpPerspective(images[anchor_idx], T, canvas_size)

        for i, img in enumerate(images):
            if i == anchor_idx: continue
            if final_homographies[i] is not None:
                warped_img = cv2.warpPerspective(img, T.dot(final_homographies[i]), canvas_size)
                mask = cv2.warpPerspective(np.full(img.shape[:2], 255, np.uint8), T.dot(final_homographies[i]), canvas_size)
                panorama = cv2.copyTo(warped_img, mask, panorama)

        return panorama

# ==============================================================================
# 🚀 FAST, INCREMENTAL STITCHER (for live preview)
# ==============================================================================
class LiveIncrementalStitcher:
    def __init__(self):
        self.finder = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.panorama = None
        self.panorama_kp = None
        self.panorama_des = None

    def add_frame(self, new_frame):
        # The first frame must be sane to start the panorama
        if self.panorama is None:
            # We already check this in the server, but as a safeguard:
            if not is_frame_sane(new_frame):
                return None
            self.panorama = new_frame
            gray_pano = cv2.cvtColor(self.panorama, cv2.COLOR_BGR2GRAY)
            self.panorama_kp, self.panorama_des = self.finder.detectAndCompute(gray_pano, None)
            return self.panorama

        gray_new = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        kp_new, des_new = self.finder.detectAndCompute(gray_new, None)

        if des_new is None or self.panorama_des is None or len(des_new) < 10: return self.panorama

        raw_matches = self.matcher.knnMatch(des_new, self.panorama_des, k=2)
        good_matches = [m for m, n in raw_matches if m.distance < 0.75 * n.distance]

        if len(good_matches) < 10: return self.panorama

        src_pts = np.float32([kp_new[m.queryIdx].pt for m in good_matches])
        dst_pts = np.float32([self.panorama_kp[m.trainIdx].pt for m in good_matches])
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)

        if H is None: return self.panorama
            
        h_new, w_new = new_frame.shape[:2]
        h_pano, w_pano = self.panorama.shape[:2]
        corners_new = np.float32([[0,0], [0,h_new], [w_new,h_new], [w_new,0]]).reshape(-1,1,2)
        corners_pano = np.float32([[0,0], [0,h_pano], [w_pano,h_pano], [w_pano,0]]).reshape(-1,1,2)
        warped_corners = cv2.perspectiveTransform(corners_new, H)
        
        all_corners = np.concatenate((corners_pano, warped_corners), axis=0)
        x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)
        
        # The transformation matrix MUST be float32 or float64 for warpPerspective.
        T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)

        canvas_size = (x_max - x_min, y_max - y_min)
        warped_new = cv2.warpPerspective(new_frame, T.dot(H), canvas_size)
        
        result = cv2.warpPerspective(self.panorama, T, canvas_size)
        
        # Simple overlay blending
        result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(result_gray, 1, 255, cv2.THRESH_BINARY_INV)[1]
        result = cv2.copyTo(warped_new, mask, result)

        self.panorama = result
        gray_pano = cv2.cvtColor(self.panorama, cv2.COLOR_BGR2GRAY)
        self.panorama_kp, self.panorama_des = self.finder.detectAndCompute(gray_pano, None)
        return self.panorama

# ==============================================================================
#  WORKFLOW FUNCTIONS
# ==============================================================================

def build_reference_panorama(source_dir, output_file="data/reference_panorama.jpg"):
    """
    STAGE 1: Stitches all images from the calibration run into a single reference panorama.
    """
    print("💡 Stage 1: Building the 360° reference panorama...")
    all_image_paths = sorted(glob.glob(os.path.join(source_dir, "*.jpg")))
    print(f"Found {len(all_image_paths)} images in {source_dir}")

    if len(all_image_paths) < 10:
        print("Error: Not enough total images to build a reference.")
        return None

    images = [cv2.imread(p) for p in all_image_paths]
    images = [img for img in images if img is not None]

    stitcher = RobustStitcher(match_confidence=0.75, ransac_thresh=4.0)
    reference_pano = stitcher.stitch(images)

    if reference_pano is not None:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        cv2.imwrite(output_file, reference_pano)
        print(f"✅ Reference panorama saved to {output_file}")
        return output_file
    else:
        print("❌ Failed to build reference panorama.")
        return None

def find_homography_to_reference(small_img, large_reference_img):
    """Helper to find the homography to map a small image onto a large reference."""
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(small_img, None)
    kp2, des2 = sift.detectAndCompute(large_reference_img, None)

    if des1 is None or des2 is None: return None
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    raw_matches = flann.knnMatch(des1, des2, k=2)

    good_matches = [m for m, n in raw_matches if m.distance < 0.75 * n.distance]
    
    if len(good_matches) < 20:
        print("Warning: Not enough matches to find a reliable homography.")
        return None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H

def calculate_placement_homographies(ref_pano_path, anchor_image_paths, output_file="data/camera_placements.npy"):
    """
    STAGE 2: Calculates the homography for each camera's anchor image to the reference.
    """
    print("\n💡 Stage 2: Calculating final camera placements...")
    reference_pano = cv2.imread(ref_pano_path)
    if reference_pano is None:
        print(f"Error: Could not load reference panorama from {ref_pano_path}")
        return None

    placements = {}
    for cam_id, anchor_path in anchor_image_paths.items():
        print(f"--- Processing anchor for camera {cam_id} ---")
        anchor_img = cv2.imread(anchor_path)
        if anchor_img is None:
            print(f"Warning: Could not load anchor image {anchor_path}")
            continue
        
        H = find_homography_to_reference(anchor_img, reference_pano)
        if H is not None:
            placements[cam_id] = H
            print(f"Successfully found placement for camera {cam_id}.")
        else:
            print(f"Failed to find placement for camera {cam_id}.")

    if len(placements) == len(anchor_image_paths):
        np.save(output_file, placements)
        print(f"\n✅ Final placement homographies saved to {output_file}")
        return output_file
    else:
        print("\n❌ Could not calculate all placements. Calibration failed.")
        return None

def stitch_from_placements(frame_paths, placements_path, ref_pano_path):
    """
    LIVE STITCH: Uses pre-calculated placements to stitch live frames.
    """
    if not os.path.exists(placements_path) or not os.path.exists(ref_pano_path):
        print("ERROR: Missing reference_panorama.jpg or camera_placements.npy")
        return None

    placements = np.load(placements_path, allow_pickle=True).item()
    ref_pano = cv2.imread(ref_pano_path)
    h, w, _ = ref_pano.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    for i, frame_path in enumerate(frame_paths):
        cam_id = str(i + 1)
        frame = cv2.imread(frame_path)
        if frame is None or cam_id not in placements:
            continue
        
        H = placements[cam_id]
        warped_frame = cv2.warpPerspective(frame, H, (w, h))
        
        # Create a mask for the warped image and place it on the canvas
        mask = cv2.warpPerspective(np.full(frame.shape[:2], 255, dtype=np.uint8), H, (w, h))
        cv2.copyTo(warped_frame, mask, canvas)
    
    stitched_path = f"data/final_stitched/capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(stitched_path, canvas)
    return stitched_path
