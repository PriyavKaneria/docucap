import cv2
import numpy as np
import time
import glob
import os
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# --- Basic Setup: Logging and Argument Parsing ---

# Configure logging for detailed output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# --- Core GPU/CPU Functions (Adapted from your code) ---
# These functions are largely the same but integrated into the new class structure.

def check_gpu_support():
    """Checks if OpenCV was built with CUDA support and a GPU is available."""
    try:
        gpu_count = cv2.cuda.getCudaEnabledDeviceCount()
        if gpu_count > 0:
            logging.info(f"✅ GPU acceleration available! Found {gpu_count} CUDA device(s).")
            return True
        else:
            logging.warning("❌ No CUDA devices found. Falling back to CPU.")
            return False
    except:
        logging.warning("❌ OpenCV not compiled with CUDA support. Falling back to CPU.")
        return False

class IncrementalStitcher:
    """
    Manages the incremental stitching process to conserve memory,
    with crash prevention and checkpointing.
    """
    # --- MODIFICATION: Added max_canvas_dim and save_interval ---
    def __init__(self, image_paths, min_match_count=15, feature_ratio=0.85, 
                 max_canvas_dim=30000, save_interval=10):
        """
        Initializes the stitcher.

        Args:
            image_paths (list): A list of paths to the images.
            min_match_count (int): Minimum number of good matches to accept a stitch.
            feature_ratio (float): Lowe's ratio test threshold.
            max_canvas_dim (int): The maximum allowed width or height for the panorama canvas.
            save_interval (int): Save a checkpoint every N successful stitches. 0 to disable.
        """
        self.image_paths = image_paths
        self.min_match_count = min_match_count
        self.feature_ratio = feature_ratio
        # --- NEW: Store new parameters ---
        self.max_canvas_dim = max_canvas_dim
        self.save_interval = save_interval

        self.panorama = None
        self.pending_paths = list(self.image_paths)
        self.skipped_paths = []

        self.gpu_available = check_gpu_support()
        self.detector = None
        self.matcher = None
        self._initialize_detectors()

    def _initialize_detectors(self):
        """Initializes GPU or CPU feature detectors and matchers."""
        if self.gpu_available:
            try:
                # Use CUDA ORB for speed
                self.detector = cv2.cuda.ORB_create(nfeatures=4000)
                self.matcher = cv2.cuda.BFMatcher(cv2.NORM_HAMMING)
                logging.info("🚀 Initialized GPU ORB detector and matcher.")
            except cv2.error as e:
                logging.error(f"⚠️ GPU detector initialization failed: {e}. Falling back to CPU.")
                self.gpu_available = False
        
        if not self.gpu_available:
            # CPU Fallback
            self.detector = cv2.ORB_create(nfeatures=2000)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
            logging.info("💻 Initialized CPU ORB detector and matcher.")

    def _detect_and_match(self, img1, img2):
        """Detects features and finds matches between two images."""
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        if self.gpu_available:
            gpu_gray1 = cv2.cuda_GpuMat()
            gpu_gray1.upload(gray1)
            gpu_gray2 = cv2.cuda_GpuMat()
            gpu_gray2.upload(gray2)
            kp1_gpu, des1_gpu = self.detector.detectAndComputeAsync(gpu_gray1, None)
            kp2_gpu, des2_gpu = self.detector.detectAndComputeAsync(gpu_gray2, None)
            kp1 = kp1_gpu.download()
            kp2 = kp2_gpu.download()
            matches_gpu = self.matcher.knnMatch(des1_gpu, des2_gpu, k=2)
            matches = cv2.cuda_BFMatcher.convert(matches_gpu)
        else:
            kp1, des1 = self.detector.detectAndCompute(gray1, None)
            kp2, des2 = self.detector.detectAndCompute(gray2, None)
            if des1 is None or des2 is None:
                logging.warning("Could not compute descriptors for one of the images.")
                return None, None, None
            matches = self.matcher.knnMatch(des1, des2, k=2)
            
        if not matches: return None, None, None
        
        good_matches = [m for m, n in matches if m.distance < self.feature_ratio * n.distance]
        return kp1, kp2, good_matches

    def _stitch_pair(self, base_img, new_img):
        """Attempts to stitch a new image to the base image (current panorama)."""
        logging.info("Finding features between current panorama and new image...")
        kp1, kp2, good_matches = self._detect_and_match(base_img, new_img)

        if not good_matches or len(good_matches) < self.min_match_count:
            logging.warning(f"Not enough good matches found ({len(good_matches) if good_matches else 0}/{self.min_match_count}). Skipping.")
            return None, None

        logging.info(f"Found {len(good_matches)} good matches. Calculating homography.")

        src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H is None:
            logging.error("Homography calculation failed. RANSAC could not find a model.")
            return None, None
            
        logging.info("Homography found successfully. Warping and blending images.")
        return H, mask

    # --- MODIFICATION: Added canvas size check ---
    def _warp_and_blend(self, base_img, new_img, H):
        """Warps the new image and blends it, checking for canvas size limits."""
        h1, w1 = base_img.shape[:2]
        h2, w2 = new_img.shape[:2]

        corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        warped_corners2 = cv2.perspectiveTransform(corners2, H)
        all_corners = np.concatenate((corners1, warped_corners2), axis=0)

        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

        canvas_width = x_max - x_min
        canvas_height = y_max - y_min

        # --- NEW: Crash prevention check ---
        if canvas_width > self.max_canvas_dim or canvas_height > self.max_canvas_dim:
            logging.error(f"DANGER: Calculated canvas size ({canvas_width}x{canvas_height}) exceeds the max dimension of {self.max_canvas_dim}.")
            logging.error("This is likely due to drift. Rejecting this stitch to prevent a crash.")
            return None # Signal failure

        logging.info(f"New canvas size will be {canvas_width}x{canvas_height}.")

        translation_dist = [-x_min, -y_min]
        H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

        # The allocation that was failing before is now protected by the check above
        warped_img2 = cv2.warpPerspective(new_img, H_translation.dot(H), (canvas_width, canvas_height))
        result_canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        result_canvas[translation_dist[1]:h1+translation_dist[1], translation_dist[0]:w1+translation_dist[0]] = base_img
        
        gray_warped = cv2.cvtColor(warped_img2, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_warped, 0, 255, cv2.THRESH_BINARY)
        result_canvas[mask > 0] = warped_img2[mask > 0]

        return result_canvas

    # --- MODIFICATION: Added checkpoint saving logic and output_filename parameter ---
    def run(self, output_filename="panorama.jpg"):
        """Executes the full incremental stitching pipeline with checkpointing."""
        if len(self.pending_paths) < 2:
            logging.error("Need at least two images to stitch.")
            return None
        
        start_time = time.time()
        output_prefix = os.path.splitext(output_filename)[0]

        initial_path = self.pending_paths.pop(0)
        logging.info(f"Setting initial panorama from: {os.path.basename(initial_path)}")
        self.panorama = cv2.imread(initial_path)
        if self.panorama is None:
            logging.critical(f"Failed to load initial image: {initial_path}")
            return None
            
        # --- NEW: Counter for successful stitches ---
        successful_stitches = 0

        # This function encapsulates the stitching logic to avoid code repetition
        def process_image_list(paths_to_process, pass_name=""):
            nonlocal successful_stitches # Use the counter from the parent scope
            for path in list(paths_to_process):
                img_name = os.path.basename(path)
                logging.info(f"--- ({pass_name}) Attempting to stitch: {img_name} ---")
                
                new_image = cv2.imread(path)
                if new_image is None:
                    logging.warning(f"Could not read {img_name}, skipping.")
                    paths_to_process.remove(path)
                    continue

                H, _ = self._stitch_pair(self.panorama, new_image)
                
                if H is not None:
                    # --- NEW: Check the return value of the safer warp_and_blend ---
                    new_panorama = self._warp_and_blend(self.panorama, new_image, H)
                    
                    if new_panorama is not None:
                        self.panorama = new_panorama
                        if path in self.pending_paths: self.pending_paths.remove(path)
                        if path in self.skipped_paths: self.skipped_paths.remove(path)
                        logging.info(f"✅ Successfully stitched {img_name} to panorama.")
                        
                        successful_stitches += 1
                        # --- NEW: Checkpoint saving logic ---
                        if self.save_interval > 0 and successful_stitches % self.save_interval == 0:
                            checkpoint_name = f"{output_prefix}_checkpoint_{successful_stitches}.jpg"
                            cv2.imwrite(checkpoint_name, self.panorama)
                            logging.info(f"📸 CHECKPOINT SAVED ({successful_stitches} stitches): {checkpoint_name}")

                    else: # Warp and blend failed the size check
                        if path not in self.skipped_paths: self.skipped_paths.append(path)
                else: # Stitch pair failed
                    if path not in self.skipped_paths: self.skipped_paths.append(path)

        # --- Run stitching passes ---
        try:
            process_image_list(self.pending_paths, pass_name="Pass 1")
            if self.skipped_paths:
                logging.info("--- Starting Second Stitching Pass on Skipped Images ---")
                process_image_list(self.skipped_paths, pass_name="Pass 2")
        except MemoryError as e:
            logging.critical(f"FATAL: A MemoryError occurred despite safeguards: {e}")
            logging.info("Attempting to save last successful panorama before exiting.")
            recovery_file = f"{output_prefix}_RECOVERY.jpg"
            cv2.imwrite(recovery_file, self.panorama)
            logging.info(f"Recovery file saved to: {recovery_file}")
            raise # Re-raise the exception after saving

        total_time = time.time() - start_time
        logging.info("--- Stitching Complete ---")
        logging.info(f"Total processing time: {total_time:.2f} seconds.")
        logging.info(f"Images stitched successfully: {successful_stitches + 1}/{len(self.image_paths)}")
        if self.skipped_paths:
            skipped_names = [os.path.basename(p) for p in self.skipped_paths]
            logging.warning(f"Discarded images: {', '.join(skipped_names)}")

        return self.panorama

# --- Main Execution Block (with new arguments) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory-efficient incremental image stitcher.")
    parser.add_argument("image_dir", type=str, help="Path to the directory containing images to stitch.")
    parser.add_argument("-o", "--output", type=str, default="panorama_stitched.jpg", help="Output file name for the panorama.")
    # --- NEW: Command-line arguments for new features ---
    parser.add_argument("--max_dim", type=int, default=30000, help="Maximum width or height of the final panorama in pixels.")
    parser.add_argument("--save_interval", type=int, default=10, help="Save a checkpoint every N stitches. Use 0 to disable.")
    args = parser.parse_args()

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(args.image_dir, ext)))
    image_paths.sort()

    if not image_paths:
        logging.critical(f"No images found in directory: {args.image_dir}")
        exit(1)
        
    logging.info(f"Found {len(image_paths)} images to process.")
    
    # Create and run the stitcher with the new arguments
    stitcher = IncrementalStitcher(
        image_paths, 
        max_canvas_dim=args.max_dim, 
        save_interval=args.save_interval
    )
    # Pass the output filename to the run method for checkpoint naming
    final_panorama = stitcher.run(output_filename=args.output)

    if final_panorama is not None:
        cv2.imwrite(args.output, final_panorama)
        logging.info(f"🎉 Final panorama successfully saved to: {args.output}")
    else:
        logging.error("Stitching process failed to produce a final result.")