import cv2
import numpy as np
import os
import glob
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import gc

def is_frame_sane(frame, min_std_dev=12.0):
    """
    Checks if a frame is likely valid and not a solid color "warm-up" frame.
    """
    if frame is None:
        return False
    if frame.shape[0] < 100 or frame.shape[1] < 100:
        return False
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    std_dev = np.std(gray)
    if std_dev < min_std_dev:
        print(f"Skipping insane frame with low detail (std dev: {std_dev:.2f})")
        return False
    return True

def smart_resize(img, target_width=800, target_height=600, maintain_aspect=True):
    """
    Intelligently resize images to prevent memory issues while maintaining quality.
    """
    h, w = img.shape[:2]
    
    if maintain_aspect:
        # Calculate scale to fit within target dimensions
        scale_w = target_width / w
        scale_h = target_height / h
        scale = min(scale_w, scale_h, 1.0)  # Don't upscale
        
        new_w = int(w * scale)
        new_h = int(h * scale)
    else:
        new_w, new_h = target_width, target_height
    
    if new_w < w or new_h < h:  # Only resize if we're making it smaller
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img

def calculate_canvas_bounds(homographies, image_shapes, max_canvas_size=15000):
    """
    Calculate optimal canvas size with safety limits.
    """
    all_corners = []
    
    for i, (H, shape) in enumerate(zip(homographies, image_shapes)):
        if H is None:
            continue
            
        h, w = shape[:2]
        corners = np.float32([[0,0], [0,h], [w,h], [w,0]]).reshape(-1,1,2)
        
        # Transform corners to panorama space
        if not np.allclose(H, np.eye(3)):
            transformed_corners = cv2.perspectiveTransform(corners, H)
        else:
            transformed_corners = corners
            
        all_corners.append(transformed_corners)
    
    if not all_corners:
        return None, None
    
    all_corners = np.concatenate(all_corners, axis=0)
    x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel())
    
    canvas_width = x_max - x_min
    canvas_height = y_max - y_min
    
    # Safety check for canvas size
    if canvas_width > max_canvas_size or canvas_height > max_canvas_size:
        print(f"WARNING: Canvas size ({canvas_width}x{canvas_height}) exceeds limit!")
        scale = min(max_canvas_size / canvas_width, max_canvas_size / canvas_height)
        canvas_width = int(canvas_width * scale)
        canvas_height = int(canvas_height * scale)
        print(f"Scaling down to: {canvas_width}x{canvas_height}")
        
        # Scale all homographies
        scale_matrix = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float32)
        homographies = [scale_matrix @ H if H is not None else None for H in homographies]
        
        x_min = int(x_min * scale)
        y_min = int(y_min * scale)
    
    return (canvas_width, canvas_height), (x_min, y_min)

class MemoryEfficientStitcher:
    def __init__(self, target_image_size=(800, 600), match_confidence=0.7, ransac_thresh=3.0):
        self.finder = cv2.SIFT_create(nfeatures=2000)  # Limit features
        self.match_confidence = match_confidence
        self.ransac_thresh = ransac_thresh
        self.target_size = target_image_size
        
    def preprocess_images(self, image_paths):
        """
        Load and preprocess images with memory management.
        """
        print(f"Preprocessing {len(image_paths)} images...")
        images = []
        
        for i, path in enumerate(image_paths):
            img = cv2.imread(path)
            if img is None:
                print(f"Warning: Could not load {path}")
                continue
                
            # Resize to manageable size
            img_resized = smart_resize(img, *self.target_size)
            
            if is_frame_sane(img_resized):
                images.append(img_resized)
                print(f"Loaded image {i+1}/{len(image_paths)}: {img_resized.shape}")
            else:
                print(f"Skipping invalid image: {path}")
                
            # Clean up original
            del img
            
        print(f"Successfully loaded {len(images)} valid images")
        return images
    
    def find_matches_batch(self, images, batch_size=5):
        """
        Find feature matches in batches to manage memory.
        """
        print("Extracting features and finding matches...")
        
        # Extract features for all images first
        keypoints = []
        descriptors = []
        
        for i, img in enumerate(images):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, des = self.finder.detectAndCompute(gray, None)
            keypoints.append(kp)
            descriptors.append(des)
            
            if (i + 1) % 10 == 0:
                print(f"Extracted features from {i+1}/{len(images)} images")
        
        # Find matches between adjacent images (for sequential stitching)
        matches = {}
        matcher = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5),
            dict(checks=50)
        )
        
        for i in range(len(images) - 1):
            if descriptors[i] is None or descriptors[i+1] is None:
                continue
                
            if len(descriptors[i]) < 10 or len(descriptors[i+1]) < 10:
                continue
                
            try:
                raw_matches = matcher.knnMatch(descriptors[i], descriptors[i+1], k=2)
                good_matches = []
                
                for match_pair in raw_matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < self.match_confidence * n.distance:
                            good_matches.append(m)
                
                if len(good_matches) > 20:
                    matches[(i, i+1)] = good_matches
                    print(f"Found {len(good_matches)} matches between images {i} and {i+1}")
                    
            except Exception as e:
                print(f"Error finding matches between {i} and {i+1}: {e}")
                continue
        
        return keypoints, descriptors, matches
    
    def calculate_homographies_sequential(self, images, keypoints, descriptors, matches):
        """
        Calculate homographies using sequential alignment.
        """
        print("Calculating homographies...")
        homographies = [np.eye(3, dtype=np.float32) for _ in images]
        
        # Start from middle image as anchor
        anchor_idx = len(images) // 2
        print(f"Using image {anchor_idx} as anchor")
        
        # Forward direction (anchor to end)
        for i in range(anchor_idx, len(images) - 1):
            pair_key = (i, i+1)
            if pair_key not in matches:
                print(f"No matches found for pair ({i}, {i+1})")
                continue
                
            good_matches = matches[pair_key]
            if len(good_matches) < 10:
                continue
                
            src_pts = np.float32([keypoints[i][m.queryIdx].pt for m in good_matches])
            dst_pts = np.float32([keypoints[i+1][m.trainIdx].pt for m in good_matches])
            
            try:
                H, mask = cv2.findHomography(
                    dst_pts, src_pts, 
                    cv2.RANSAC, 
                    self.ransac_thresh,
                    maxIters=2000
                )
                
                if H is not None:
                    # Accumulate transformation
                    homographies[i+1] = homographies[i] @ H
                    inliers = np.sum(mask)
                    print(f"Image {i+1}: {inliers}/{len(good_matches)} inliers")
                else:
                    print(f"Failed to find homography for image {i+1}")
                    
            except Exception as e:
                print(f"Error calculating homography for image {i+1}: {e}")
        
        # Backward direction (anchor to start)
        for i in range(anchor_idx, 0, -1):
            pair_key = (i-1, i)
            if pair_key not in matches:
                continue
                
            good_matches = matches[pair_key]
            if len(good_matches) < 10:
                continue
                
            src_pts = np.float32([keypoints[i-1][m.queryIdx].pt for m in good_matches])
            dst_pts = np.float32([keypoints[i][m.trainIdx].pt for m in good_matches])
            
            try:
                H, mask = cv2.findHomography(
                    src_pts, dst_pts,
                    cv2.RANSAC,
                    self.ransac_thresh,
                    maxIters=2000
                )
                
                if H is not None:
                    homographies[i-1] = homographies[i] @ H
                    inliers = np.sum(mask)
                    print(f"Image {i-1}: {inliers}/{len(good_matches)} inliers")
                    
            except Exception as e:
                print(f"Error calculating homography for image {i-1}: {e}")
        
        return homographies
    
    def stitch_with_blending(self, images, homographies, max_canvas_size=12000):
        """
        Stitch images with proper blending and memory management.
        """
        print("Stitching images...")
        
        # Calculate canvas size
        canvas_size, offset = calculate_canvas_bounds(
            homographies, 
            [img.shape for img in images], 
            max_canvas_size
        )
        
        if canvas_size is None:
            print("Error: Could not calculate canvas bounds")
            return None
            
        canvas_width, canvas_height = canvas_size
        x_offset, y_offset = offset
        
        print(f"Canvas size: {canvas_width} x {canvas_height}")
        
        # Create translation matrix
        T = np.array([
            [1, 0, -x_offset],
            [0, 1, -y_offset],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Initialize canvas
        panorama = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        weight_sum = np.zeros((canvas_height, canvas_width), dtype=np.float32)
        
        # Warp and blend each image
        for i, (img, H) in enumerate(zip(images, homographies)):
            if H is None:
                continue
                
            print(f"Warping image {i+1}/{len(images)}")
            
            try:
                # Apply translation to homography
                final_H = T @ H
                
                # Warp image
                warped = cv2.warpPerspective(img, final_H, (canvas_width, canvas_height))
                
                # Create weight mask (distance from center)
                h, w = img.shape[:2]
                center_x, center_y = w // 2, h // 2
                y_coords, x_coords = np.ogrid[:h, :w]
                weight_mask = 1.0 / (1.0 + 0.1 * ((x_coords - center_x)**2 + (y_coords - center_y)**2)**0.5)
                
                # Warp weight mask
                warped_weights = cv2.warpPerspective(weight_mask, final_H, (canvas_width, canvas_height))
                
                # Find valid regions
                valid_mask = (warped.sum(axis=2) > 0) & (warped_weights > 0)
                
                if np.any(valid_mask):
                    # Weighted blending
                    for c in range(3):
                        panorama[:,:,c] = np.where(
                            valid_mask,
                            (panorama[:,:,c] * weight_sum + warped[:,:,c] * warped_weights) / 
                            (weight_sum + warped_weights + 1e-10),
                            panorama[:,:,c]
                        )
                    
                    weight_sum = np.where(valid_mask, weight_sum + warped_weights, weight_sum)
                
                # Memory cleanup
                del warped, warped_weights, valid_mask
                gc.collect()
                
            except Exception as e:
                print(f"Error warping image {i}: {e}")
                continue
        
        return panorama
    
    def stitch(self, image_paths):
        """
        Main stitching function with memory management.
        """
        try:
            # Step 1: Load and preprocess images
            images = self.preprocess_images(image_paths)
            if len(images) < 2:
                print("Error: Not enough valid images to stitch")
                return None
            
            # Step 2: Find feature matches
            keypoints, descriptors, matches = self.find_matches_batch(images)
            
            if not matches:
                print("Error: No matches found between images")
                return None
            
            # Step 3: Calculate homographies
            homographies = self.calculate_homographies_sequential(
                images, keypoints, descriptors, matches
            )
            
            # Step 4: Stitch with blending
            panorama = self.stitch_with_blending(images, homographies)
            
            # Cleanup
            del images, keypoints, descriptors, matches
            gc.collect()
            
            return panorama
            
        except Exception as e:
            print(f"Error in stitching process: {e}")
            return None

def build_reference_panorama_efficient(source_dir, output_file="data/reference_panorama.jpg", 
                                     target_size=(800, 600)):
    """
    Memory-efficient panorama building.
    """
    print("💡 Building reference panorama with memory management...")
    
    all_image_paths = sorted(glob.glob(os.path.join(source_dir, "*.jpg")))
    print(f"Found {len(all_image_paths)} images")
    
    if len(all_image_paths) < 10:
        print("Error: Need at least 10 images")
        return None
    
    # Use memory-efficient stitcher
    stitcher = MemoryEfficientStitcher(
        target_image_size=target_size,
        match_confidence=0.75,
        ransac_thresh=4.0
    )
    
    panorama = stitcher.stitch(all_image_paths)
    
    if panorama is not None:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        cv2.imwrite(output_file, panorama)
        print(f"✅ Reference panorama saved to {output_file}")
        print(f"Final size: {panorama.shape}")
        return output_file
    else:
        print("❌ Failed to build reference panorama")
        return None

# Example usage
if __name__ == "__main__":
    # Build panorama from 50+ images
    result = build_reference_panorama_efficient(
        source_dir="path/to/your/images",
        output_file="data/panorama_50plus.jpg",
        target_size=(1024, 768)  # Adjust based on your memory constraints
    )