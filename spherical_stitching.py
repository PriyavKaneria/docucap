import cv2
import numpy as np
import os
import argparse
import sys
import glob

class SimpleSphericalStitcher:
    """
    A stitcher that bypasses the problematic cv2.detail module entirely.
    Uses direct OpenCV Stitcher class which handles feature detection internally.
    """
    def __init__(self, image_paths):
        self.image_paths = image_paths
        
        # Try different stitcher modes
        try:
            # PANORAMA mode for general panoramic stitching
            self.stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
            print("Using PANORAMA mode stitcher")
        except:
            try:
                # SCANS mode is good for ordered image sequences (like panoramic sweeps)
                self.stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)
                print("Using SCANS mode stitcher")
            except:
                # Fallback for older OpenCV versions
                self.stitcher = cv2.Stitcher_create()
                print("Using default stitcher")

    def stitch(self, output_path="panorama.jpg"):
        """Main method to execute the stitching pipeline."""
        
        print("--- Loading Images ---")
        images = self._load_images()
        if not images:
            return None
        
        print(f"Loaded {len(images)} images")
        
        print("--- Stitching (this may take a while) ---")
        # The OpenCV Stitcher handles everything internally
        status, panorama = self.stitcher.stitch(images)
        
        if status == cv2.Stitcher_OK:
            print("--- Stitching Complete ---")
            return panorama
        else:
            error_messages = {
                cv2.Stitcher_ERR_NEED_MORE_IMGS: "Need more images",
                cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Homography estimation failed", 
                cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Camera parameter adjustment failed"
            }
            error_msg = error_messages.get(status, f"Unknown error (code: {status})")
            print(f"Stitching failed: {error_msg}")
            return None

    def _load_images(self):
        images = []
        for path in self.image_paths:
            img = cv2.imread(path)
            if img is None:
                print(f"Failed to load image: {path}. Skipping.")
                continue
            images.append(img)
        
        if not images:
            print("No images were loaded successfully.")
            return None
        return images


class ManualSphericalStitcher:
    """
    Manual implementation that completely avoids cv2.detail module.
    Uses basic feature matching and homography estimation.
    """
    def __init__(self, image_paths):
        self.image_paths = image_paths
        # Use ORB for reliability
        self.detector = cv2.ORB_create(nfeatures=5000)
        # Use FLANN matcher for speed
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                           table_number=6,
                           key_size=12,
                           multi_probe_level=1)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def stitch(self, output_path="panorama.jpg"):
        print("--- Loading Images ---")
        images = self._load_images()
        if not images or len(images) < 2:
            return None
        
        print("--- Finding and Matching Features ---")
        # Find features in all images
        keypoints_list = []
        descriptors_list = []
        
        for i, img in enumerate(images):
            print(f"Processing image {i+1}/{len(images)}")
            kp, desc = self.detector.detectAndCompute(img, None)
            if desc is not None and len(desc) > 10:  # Need minimum features
                keypoints_list.append(kp)
                descriptors_list.append(desc)
            else:
                print(f"Not enough features in image {i+1}, skipping")
                
        if len(descriptors_list) < 2:
            print("Not enough images with sufficient features")
            return None
            
        print("--- Sequential Stitching ---")
        # Start with first image as base
        result = images[0].copy()
        
        for i in range(1, len(descriptors_list)):
            print(f"Stitching image {i+1} to panorama...")
            
            # Match features between current result and next image
            matches = self.matcher.knnMatch(descriptors_list[0], descriptors_list[i], k=2)
            
            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) < 10:
                print(f"Not enough good matches for image {i+1}")
                continue
                
            # Extract matched keypoints
            src_pts = np.float32([keypoints_list[0][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints_list[i][m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Find homography
            H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            
            if H is None:
                print(f"Could not find homography for image {i+1}")
                continue
                
            # Warp the next image
            h1, w1 = result.shape[:2]
            h2, w2 = images[i].shape[:2]
            
            # Calculate output size
            corners = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
            transformed_corners = cv2.perspectiveTransform(corners, H)
            
            all_corners = np.concatenate([np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2), 
                                        transformed_corners], axis=0)
            
            [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel())
            [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel())
            
            # Translation matrix
            translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)
            
            # Warp images
            output_size = (x_max - x_min, y_max - y_min)
            warped_img = cv2.warpPerspective(images[i], translation @ H, output_size)
            warped_result = cv2.warpPerspective(result, translation, output_size)
            
            # Simple blending (take max where both exist)
            mask_result = (warped_result.sum(axis=2) > 0).astype(np.uint8)
            mask_img = (warped_img.sum(axis=2) > 0).astype(np.uint8)
            
            # Blend overlapping regions
            overlap = mask_result & mask_img
            result = np.where(overlap[..., None], 
                            (warped_result.astype(np.float32) + warped_img.astype(np.float32)) / 2,
                            warped_result + warped_img).astype(np.uint8)
            
        print("--- Manual Stitching Complete ---")
        return result

    def _load_images(self):
        images = []
        for path in self.image_paths:
            img = cv2.imread(path)
            if img is None:
                print(f"Failed to load image: {path}. Skipping.")
                continue
            images.append(img)
        
        if not images:
            print("No images were loaded successfully.")
            return None
        return images


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bypass detail module stitcher.")
    parser.add_argument("image_dir", type=str, help="Path to the directory containing images to stitch.")
    parser.add_argument("-o", "--output", type=str, default="panorama_bypass.jpg", help="Output file name.")
    parser.add_argument("--method", choices=['simple', 'manual'], default='simple',
                        help="Stitching method: 'simple' uses cv2.Stitcher, 'manual' implements custom stitching")
    parser.add_argument("--order", type=str, default="esp3,esp2,esp1",
                        help="Comma-separated list of camera IDs in the desired stitching order (e.g., 'esp3,esp2,esp1').")
    args = parser.parse_args()

    # --- Custom Sorting Logic ---
    # Build a priority map from the --order argument
    camera_order_list = [item.strip() for item in args.order.split(',')]
    camera_order_map = {cam_id: index for index, cam_id in enumerate(camera_order_list)}

    def get_sort_key(image_path):
        """
        Parses the image filename to determine its sort order.
        Sorts by camera ID based on the --order argument, then by timestamp.
        """
        import re
        filename = os.path.basename(image_path)
        # Regex to extract camera number and timestamp from filenames like 'esp_3_frame_12345.678.jpg'
        match = re.search(r'esp_(\d+)_frame_([\d.]+)\.jpg', filename)
        if match:
            camera_num = int(match.group(1))
            cam_id = f"esp{camera_num}" # Create ID like 'esp3'
            timestamp = float(match.group(2))
            
            # Get priority, default to a high number if camera not in the order list
            priority = camera_order_map.get(cam_id, 999)
            return (priority, timestamp)
        
        # Return a default tuple for files that don't match the pattern
        return (999, 0)

    # --- Image Loading and Sorting ---
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(args.image_dir, ext)))

    # Apply the custom sorting function
    image_paths.sort(key=get_sort_key)

    if not image_paths:
        print(f"No images found in directory: {args.image_dir}")
        sys.exit(1)

    print(f"Found {len(image_paths)} images to process.")
    print("--- Sorted Image Order ---")
    for path in image_paths:
        print(f"  - {os.path.basename(path)}")
    print("--------------------------")
    
    print(f"OpenCV version: {cv2.__version__}")

    if args.method == 'simple':
        stitcher = SimpleSphericalStitcher(image_paths)
    else:
        stitcher = ManualSphericalStitcher(image_paths)

    final_panorama = stitcher.stitch(output_path=args.output)

    if final_panorama is not None:
        cv2.imwrite(args.output, final_panorama)
        print(f"🎉 Panorama successfully saved to: {args.output}")
    else:
        print("Stitching process failed to produce a final result.")