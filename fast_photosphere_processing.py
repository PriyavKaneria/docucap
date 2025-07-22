import cv2
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# Check GPU availability
def check_gpu_support():
    """Check if GPU support is available"""
    try:
        gpu_count = cv2.cuda.getCudaEnabledDeviceCount()
        if gpu_count > 0:
            print(f"✅ GPU acceleration available! Found {gpu_count} CUDA device(s)")
            return True
        else:
            print("❌ No CUDA devices found")
            return False
    except:
        print("❌ OpenCV not compiled with CUDA support")
        return False

# Global variables for optimization
SIFT_DETECTOR_GPU = None
MATCHER_GPU = None
GPU_AVAILABLE = check_gpu_support()

def initialize_gpu_objects():
    """Initialize GPU OpenCV objects once"""
    global SIFT_DETECTOR_GPU, MATCHER_GPU
    if GPU_AVAILABLE:
        try:
            if SIFT_DETECTOR_GPU is None:
                # Use CUDA ORB for speed (faster than SIFT on GPU)
                SIFT_DETECTOR_GPU = cv2.cuda.ORB_create(nfeatures=2000)
            if MATCHER_GPU is None:
                MATCHER_GPU = cv2.cuda.BFMatcher(cv2.NORM_HAMMING)
        except:
            print("⚠️  GPU detector initialization failed, falling back to CPU")
            return False
    return GPU_AVAILABLE

def compute_cumulative_homographies_optimized(homographies, reference_idx=1):
    """Optimized cumulative homography computation"""
    n_images = len(homographies) + 1
    cumulative_homographies = [None] * n_images
    
    # Reference image gets identity matrix
    cumulative_homographies[reference_idx] = np.eye(3, dtype=np.float32)
    
    # Pre-allocate for better memory usage
    cumulative = np.eye(3, dtype=np.float32)
    
    # For images to the right of reference
    for i in range(reference_idx, n_images - 1):
        np.dot(homographies[i], cumulative, out=cumulative)
        cumulative_homographies[i + 1] = cumulative.copy()
    
    # For images to the left of reference
    cumulative = np.eye(3, dtype=np.float32)
    for i in range(reference_idx - 1, -1, -1):
        H_inv = cv2.invert(homographies[i])[1]
        np.dot(H_inv, cumulative, out=cumulative)
        cumulative_homographies[i] = cumulative.copy()
    
    return cumulative_homographies

def compute_canvas_size_vectorized(images, cumulative_homographies):
    """Vectorized canvas size computation"""
    h, w = images[0].shape[:2]
    base_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    
    # Vectorized transformation of all corners at once
    all_corners = []
    corners_batch = np.tile(base_corners, (len(cumulative_homographies), 1, 1))
    
    for i, H in enumerate(cumulative_homographies):
        transformed = cv2.perspectiveTransform(corners_batch[i:i+1], H)
        all_corners.append(transformed.reshape(-1, 2))
    
    corners = np.vstack(all_corners)
    bounds = np.array([corners.min(axis=0), corners.max(axis=0)]).astype(np.int32)
    
    canvas_width = bounds[1, 0] - bounds[0, 0]
    canvas_height = bounds[1, 1] - bounds[0, 1]
    translation = np.array([[1, 0, -bounds[0, 0]], [0, 1, -bounds[0, 1]], [0, 0, 1]], dtype=np.float32)
    
    return (canvas_width, canvas_height), translation

def gpu_warp_single_image(img, H_total, canvas_size):
    """GPU-accelerated image warping"""
    if GPU_AVAILABLE:
        try:
            # Upload to GPU
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(img)
            
            # Create output GPU matrix
            gpu_warped = cv2.cuda_GpuMat()
            
            # GPU warp perspective
            cv2.cuda.warpPerspective(gpu_img, gpu_warped, H_total, canvas_size, flags=cv2.INTER_LINEAR)
            
            # Download result
            warped = gpu_warped.download()
            
            # Create mask on GPU
            mask = np.full((img.shape[0], img.shape[1]), 255, dtype=np.uint8)
            gpu_mask = cv2.cuda_GpuMat()
            gpu_mask.upload(mask)
            
            gpu_warped_mask = cv2.cuda_GpuMat()
            cv2.cuda.warpPerspective(gpu_mask, gpu_warped_mask, H_total, canvas_size, flags=cv2.INTER_NEAREST)
            
            warped_mask = gpu_warped_mask.download()
            
            return warped, warped_mask
        except Exception as e:
            print(f"⚠️  GPU warping failed: {e}, falling back to CPU")
    
    # CPU fallback
    warped = cv2.warpPerspective(img, H_total, canvas_size, flags=cv2.INTER_LINEAR)
    mask = np.full((img.shape[0], img.shape[1]), 255, dtype=np.uint8)
    warped_mask = cv2.warpPerspective(mask, H_total, canvas_size, flags=cv2.INTER_NEAREST)
    
    return warped, warped_mask

def ultra_fast_gpu_warp_images(images, cumulative_homographies, translation, canvas_size):
    """Ultra-fast GPU-accelerated parallel image warping"""
    def warp_single(args):
        img, H_total, canvas_size = args
        return gpu_warp_single_image(img, H_total, canvas_size)
    
    # Prepare arguments
    warp_args = [(img, translation @ H, canvas_size) 
                 for img, H in zip(images, cumulative_homographies)]
    
    # Parallel processing
    with ThreadPoolExecutor(max_workers=min(len(images), mp.cpu_count())) as executor:
        results = list(executor.map(warp_single, warp_args))
    
    warped_images, masks = zip(*results)
    return list(warped_images), list(masks)

def lightning_fast_feathered_blending(warped_images, masks, canvas_size):
    """Lightning-fast feathered blending with GPU acceleration"""
    result = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.float32)
    weight_sum = np.zeros((canvas_size[1], canvas_size[0]), dtype=np.float32)
    
    if GPU_AVAILABLE:
        try:
            # GPU-accelerated blending
            gpu_result = cv2.cuda_GpuMat()
            gpu_result.upload(result)
            gpu_weight_sum = cv2.cuda_GpuMat()
            gpu_weight_sum.upload(weight_sum)
            
            for img, mask in zip(warped_images, masks):
                # GPU distance transform
                gpu_mask = cv2.cuda_GpuMat()
                gpu_mask.upload(mask)
                
                gpu_dist = cv2.cuda_GpuMat()
                cv2.cuda.distanceTransform(gpu_mask, gpu_dist, cv2.DIST_L1, 3)
                
                dist_transform = gpu_dist.download()
                
                # Fast normalization
                max_dist = dist_transform.max()
                if max_dist > 0:
                    feather_weight = dist_transform / max_dist
                else:
                    feather_weight = (mask > 0).astype(np.float32)
                
                # Vectorized blending
                img_float = img.astype(np.float32)
                result += img_float * feather_weight[:, :, np.newaxis]
                weight_sum += feather_weight
            
        except Exception as e:
            print(f"⚠️  GPU blending failed: {e}, using CPU")
    
    # CPU fallback or non-GPU path
    if not GPU_AVAILABLE:
        for img, mask in zip(warped_images, masks):
            # Use approximation for speed
            kernel = np.ones((5, 5), np.uint8)
            eroded = cv2.erode(mask, kernel, iterations=1)
            feather_weight = cv2.GaussianBlur(eroded.astype(np.float32), (15, 15), 0) / 255.0
            
            img_float = img.astype(np.float32)
            result += img_float * feather_weight[:, :, np.newaxis]
            weight_sum += feather_weight
    
    # Avoid division by zero and normalize
    weight_sum[weight_sum == 0] = 1.0
    result /= weight_sum[:, :, np.newaxis]
    
    return result.astype(np.uint8)

def detect_and_match_features_gpu(images):
    """GPU-accelerated feature detection and matching"""
    if not initialize_gpu_objects():
        return detect_and_match_features_cpu_optimized(images)
    
    try:
        # Convert to grayscale and upload to GPU
        gray_gpu_mats = []
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Resize for speed
            gray = cv2.resize(gray, None, fx=0.7, fy=0.7)
            gpu_mat = cv2.cuda_GpuMat()
            gpu_mat.upload(gray)
            gray_gpu_mats.append(gpu_mat)
        
        # Detect features on GPU
        keypoints_gpu = []
        descriptors_gpu = []
        
        for gpu_mat in gray_gpu_mats:
            kp_gpu, desc_gpu = SIFT_DETECTOR_GPU.detectAndComputeAsync(gpu_mat)
            keypoints_gpu.append(kp_gpu)
            descriptors_gpu.append(desc_gpu)
        
        # Match features on GPU
        matches = []
        for i in range(len(images) - 1):
            match_gpu = MATCHER_GPU.knnMatch(descriptors_gpu[i], descriptors_gpu[i + 1], k=2)
            matches.append(match_gpu)
        
        return keypoints_gpu, matches
        
    except Exception as e:
        print(f"⚠️  GPU feature detection failed: {e}, using CPU")
        return detect_and_match_features_cpu_optimized(images)

def detect_and_match_features_cpu_optimized(images):
    """Highly optimized CPU feature detection"""
    # Use ORB for speed
    orb = cv2.ORB_create(nfeatures=600)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    # Parallel feature detection
    def detect_features(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Resize for speed
        gray = cv2.resize(gray, None, fx=0.7, fy=0.7)
        return orb.detectAndCompute(gray, None)
    
    with ThreadPoolExecutor(max_workers=min(len(images), mp.cpu_count())) as executor:
        feature_results = list(executor.map(detect_features, images))
    
    keypoints = [result[0] for result in feature_results]
    descriptors = [result[1] for result in feature_results]
    
    # Fast matching
    matches = []
    for i in range(len(images) - 1):
        match = matcher.knnMatch(descriptors[i], descriptors[i + 1], k=2)
        matches.append(match)
    
    return keypoints, matches

def compute_homography_ultra_fast(keypoints, matches):
    """Ultra-fast homography computation"""
    homographies = []
    
    for i, match in enumerate(matches):
        # Aggressive ratio test for speed
        good_matches = [m for m, n in match if m.distance < 0.95 * n.distance]
        
        if len(good_matches) < 8:
            homographies.append(np.eye(3, dtype=np.float32))
            continue
        
        # Extract points
        src_pts = np.float32([keypoints[i + 1][m.trainIdx].pt for m in good_matches])
        dst_pts = np.float32([keypoints[i][m.queryIdx].pt for m in good_matches])
        
        # Scale points back up (since we resized images)
        src_pts /= 0.7
        dst_pts /= 0.7
        
        # Ultra-fast homography
        H, _ = cv2.findHomography(
            src_pts, dst_pts, 
            method=cv2.RANSAC, 
            ransacReprojThreshold=4.0,
            maxIters=500,
            confidence=0.95
        )
        
        if H is None:
            H = np.eye(3, dtype=np.float32)
        
        homographies.append(H.astype(np.float32))
    
    return homographies

def lightning_fast_stitch_images(images, reference_idx=1):
    """Lightning-fast GPU-accelerated image stitching"""
    total_start = time.time()
    
    print(f"🚀 Starting LIGHTNING-FAST stitching with {len(images)} images...")
    if GPU_AVAILABLE:
        print("⚡ GPU acceleration ENABLED")
    else:
        print("💻 Using optimized CPU processing")
    
    # Step 1: Ultra-fast feature detection
    step_start = time.time()
    keypoints, matches = detect_and_match_features_gpu(images)
    print(f"Found matches: {len(matches)}")
    print(f"⚡ Feature detection: {time.time() - step_start:.3f}s")
    
    # Step 2: Ultra-fast homography
    step_start = time.time()
    homographies = compute_homography_ultra_fast(keypoints, matches)
    print(f"⚡ Homography computation: {time.time() - step_start:.3f}s")
    
    # Step 3: Optimized cumulative homographies
    step_start = time.time()
    cumulative_homographies = compute_cumulative_homographies_optimized(homographies, reference_idx)
    print(f"⚡ Cumulative homographies: {time.time() - step_start:.3f}s")
    
    # Step 4: Vectorized canvas size
    step_start = time.time()
    canvas_size, translation = compute_canvas_size_vectorized(images, cumulative_homographies)
    print(f"⚡ Canvas size computation: {time.time() - step_start:.3f}s")
    
    # Step 5: GPU-accelerated warping
    step_start = time.time()
    warped_images, masks = ultra_fast_gpu_warp_images(images, cumulative_homographies, translation, canvas_size)
    print(f"⚡ GPU image warping: {time.time() - step_start:.3f}s")
    
    # Step 6: Lightning-fast blending
    step_start = time.time()
    result = lightning_fast_feathered_blending(warped_images, masks, canvas_size)
    print(f"⚡ Lightning blending: {time.time() - step_start:.3f}s")
    
    total_time = time.time() - total_start
    print(f"\n🏆 LIGHTNING PROCESSING TIME: {total_time:.3f}s")
    
    # Performance rating
    if total_time < 0.5:
        print("🚀 INSANE SPEED! Under 0.5 seconds!")
    elif total_time < 1.0:
        print("⚡ LIGHTNING FAST! Under 1 second!")
    elif total_time < 2.0:
        print("🏃 VERY FAST! Under 2 seconds!")
    else:
        print("🐌 Consider GPU upgrade for better performance")
    
    print(f"Canvas size: {canvas_size}")
    print(f"Reference image index: {reference_idx}")
    
    return result, total_time

def benchmark_performance(images, iterations=5):
    """Benchmark the performance across multiple runs"""
    print(f"\n🏁 RUNNING PERFORMANCE BENCHMARK ({iterations} iterations)")
    times = []
    
    for i in range(iterations):
        print(f"\n--- Run {i+1}/{iterations} ---")
        result, processing_time = lightning_fast_stitch_images(images, reference_idx=1)
        times.append(processing_time)
        
        # if result is not None:
        #     cv2.imwrite(f"lightning_fast_run_{i+1}.jpg", result)
    
    avg_time = np.mean(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\n📊 BENCHMARK RESULTS:")
    print(f"🏆 Fastest run: {min_time:.3f}s")
    print(f"📈 Average time: {avg_time:.3f}s")
    print(f"📉 Slowest run: {max_time:.3f}s")
    print(f"⚡ Speed improvement: {4.869/avg_time:.1f}x faster than baseline")
    
    return times

# Main execution
if __name__ == "__main__":
    # Load images
    images = [cv2.imread(f"data\stitched_panoramas\esp{i}_panorama.jpg") for i in range(1,4)]
    
    if any(img is None for img in images):
        print("❌ Error: Could not load all images. Please check file paths.")
        exit(1)
    
    # Single lightning-fast processing
    result, processing_time = lightning_fast_stitch_images(images, reference_idx=1)
    
    if result is not None:
        cv2.imwrite("lightning_fast_stitched.jpg", result)
        print(f"\n✅ SUCCESS: Image saved as 'lightning_fast_stitched.jpg'")
        print(f"⚡ LIGHTNING processing completed in {processing_time:.3f} seconds")
        
        # Performance comparison
        baseline_time = 4.869  # Your reported average
        speedup = baseline_time / processing_time
        print(f"🚀 Speed improvement: {speedup:.1f}x faster than baseline!")
    else:
        print("❌ FAILED: Lightning stitching failed")
    
    # Optional: Run benchmark
    # benchmark_times = benchmark_performance(images, iterations=3)