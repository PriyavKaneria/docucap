#!/usr/bin/env python3
"""
Complete Panorama Pipeline (N-camera, incl. 4 ESP support)
- Generate a reference panorama from many frames (rotation sweep)
- Extract multi-camera calibration (N images -> panorama)
- Stitch new N-camera captures using saved calibration

Usage examples:
  # Full pipeline from a capture directory (uses index per ESP for anchors)
  python panorama_pipeline.py --mode full --input_dir calibration_run_min --index 5

  # Stitch only with an existing calibration and N images
  python panorama_pipeline.py --mode stitch --calibration_file multi_calibration/latest.npz --images cam1.jpg cam2.jpg cam3.jpg cam4.jpg

Notes:
- This version generalizes the earlier "triplet" flow to N cameras (default 4).
- Output directories are placed relative to this file's directory (final/...)
"""

import os
import sys
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union

import numpy as np
import cv2 as cv
from datetime import datetime
from pathlib import Path

# Stitching library imports
from stitching.images import Images
from stitching.feature_detector import FeatureDetector
from stitching.feature_matcher import FeatureMatcher
from stitching.camera_estimator import CameraEstimator
from stitching.camera_adjuster import CameraAdjuster
from stitching.camera_wave_corrector import WaveCorrector
from stitching.warper import Warper
from stitching.seam_finder import SeamFinder
from stitching.exposure_error_compensator import ExposureErrorCompensator
from stitching.blender import Blender


############################ CONFIGURATION ############################
class PipelineConfig:
    """Configuration parameters for the panorama pipeline"""
    BASE_DIR = Path(__file__).parent

    # Feature detection
    DETECTOR_TYPE = "orb"  # orb, sift, brisk, akaze
    N_FEATURES = 5000

    # Feature matching
    MATCHER_TYPE = "homography"
    CONFIDENCE_THRESHOLD = 0.5

    # Output directories (absolute inside final/)
    OUTPUT_DIR = str(BASE_DIR / "pipeline_outputs")
    CALIBRATION_DIR = str(BASE_DIR / "calibration_params")
    MULTI_CALIBRATION_DIR = str(BASE_DIR / "multi_calibration")

    # Image processing options
    PERFORM_EXPOSURE_COMPENSATION = True
    PERFORM_SEAM_FINDING = True

    # ESP camera naming pattern (files should contain these substrings)
    # Default to 4 cameras
    ESP_CAMERAS = ["esp_1", "esp_2", "esp_3", "esp_4"]


############################ PIPELINE CLASS ############################
class PanoramaPipeline:
    """Complete panorama generation and stitching pipeline for N cameras"""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.logger = self._setup_logger()

        # Pipeline state
        self.panorama_path: Optional[str] = None
        self.calibration_file: Optional[str] = None
        self.anchor_images: Optional[List[str]] = None

        # Ensure output directories exist
        for dir_name in [
            self.config.OUTPUT_DIR,
            self.config.CALIBRATION_DIR,
            self.config.MULTI_CALIBRATION_DIR,
        ]:
            os.makedirs(dir_name, exist_ok=True)

    def _setup_logger(self):
        """Setup logging for the pipeline"""
        import logging

        logger = logging.getLogger("PanoramaPipeline")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    ############################ STEP 1: PANORAMA GENERATION ############################
    def step1_generate_panorama(self, input_dir: str) -> str:
        """
        Step 1: Generate reference panorama from a directory of frames.

        Args:
            input_dir: Directory containing many images (rotation sweep)

        Returns:
            Path to generated panorama (PNG)
        """
        self.logger.info("=" * 60)
        self.logger.info("STEP 1: GENERATING PANORAMA (REFERENCE)")
        self.logger.info("=" * 60)

        # Load images from dir
        self.logger.info(f"Loading images from: {input_dir}")
        orig_imgs = [
            os.path.join(input_dir, f)
            for f in sorted(os.listdir(input_dir))
            if f != ".DS_Store" and f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if not orig_imgs:
            raise ValueError(f"No images found in {input_dir}")

        self.logger.info(f"Found {len(orig_imgs)} images")

        # Resize images
        self.logger.info("Resizing images...")
        images = Images.of(orig_imgs)

        medium_imgs = list(images.resize(Images.Resolution.MEDIUM))
        low_imgs = list(images.resize(Images.Resolution.LOW))
        final_imgs = list(images.resize(Images.Resolution.FINAL))

        # Log image sizes
        image_sizes = {
            "original": images.sizes[0],
            "medium": images.get_image_size(medium_imgs[0]),
            "low": images.get_image_size(low_imgs[0]),
            "final": images.get_image_size(final_imgs[0]),
        }

        for size_name, size in image_sizes.items():
            pixel_count = int(np.prod(size))
            self.logger.info(f"{size_name.title()} Size: {size} -> {pixel_count:,} px")

        # Feature detection
        self.logger.info("Detecting features...")
        finder = FeatureDetector(
            detector=self.config.DETECTOR_TYPE, nfeatures=self.config.N_FEATURES
        )
        features = [finder.detect_features(img) for img in medium_imgs]
        self.logger.info(
            f"Features detected per image: {[len(f.getKeypoints()) for f in features]}"
        )

        # Feature matching
        self.logger.info("Matching features...")
        matcher = FeatureMatcher(matcher_type=self.config.MATCHER_TYPE)
        matches = matcher.match_features(features)

        conf_matrix = matcher.get_confidence_matrix(matches)
        self.logger.info("Feature matching confidence matrix:")
        self.logger.info(f"\n{conf_matrix}")

        conf_thres = float(
            np.min(np.append(np.diag(conf_matrix, k=1), conf_matrix[-1][0]))
        )
        self.logger.info(f"Using confidence threshold: {conf_thres:.3f}")

        all_relevant = matcher.draw_matches_matrix(
            medium_imgs,
            features,
            matches,
            conf_thresh=conf_thres,
            inliers=True,
            matchColor=(0, 255, 0),
        )
        self.logger.info("Relevant matches between neighbors:")
        for idx1, idx2, _ in all_relevant:
            n = len(medium_imgs)
            if idx2 in [(idx1 + 1) % n, (idx1 - 1) % n]:
                self.logger.info(f"  Matched Image {idx1+1} to Image {idx2+1}")

        # Estimate / adjust cameras
        self.logger.info("Estimating camera parameters...")
        camera_estimator = CameraEstimator()
        camera_adjuster = CameraAdjuster()
        wave_corrector = WaveCorrector()

        cameras = camera_estimator.estimate(features, matches)
        cameras = camera_adjuster.adjust(features, matches, cameras)
        cameras = wave_corrector.correct(cameras)
        self.logger.info(f"Estimated {len(cameras)} cameras")

        # Warp setup
        self.logger.info("Setting up warper...")
        warper = Warper()
        warper.set_scale(cameras)
        warper_scale = warper.scale

        # Save full calibration (for reference)
        cal_npz, cal_json = self._save_full_calibration(
            cameras, warper_scale, image_sizes
        )
        self.logger.info(f"Full calibration saved: {cal_npz}")

        # Warp and blend
        self.logger.info("Warping low resolution images...")
        low_sizes = images.get_scaled_img_sizes(Images.Resolution.LOW)
        aspect_low = images.get_ratio(Images.Resolution.MEDIUM, Images.Resolution.LOW)
        warped_low_imgs = list(warper.warp_images(low_imgs, cameras, aspect_low))
        warped_low_masks = list(
            warper.create_and_warp_masks(low_sizes, cameras, aspect_low)
        )
        low_corners, low_sizes = warper.warp_rois(low_sizes, cameras, aspect_low)

        self.logger.info("Warping final resolution images...")
        final_sizes = images.get_scaled_img_sizes(Images.Resolution.FINAL)
        aspect_final = images.get_ratio(Images.Resolution.MEDIUM, Images.Resolution.FINAL)
        warped_final_imgs = list(warper.warp_images(final_imgs, cameras, aspect_final))
        warped_final_masks = list(
            warper.create_and_warp_masks(final_sizes, cameras, aspect_final)
        )
        final_corners, final_sizes = warper.warp_rois(
            final_sizes, cameras, aspect_final
        )

        # Seam finding
        if self.config.PERFORM_SEAM_FINDING:
            self.logger.info("Finding optimal seams...")
            seam_finder = SeamFinder()
            seam_masks = seam_finder.find(warped_low_imgs, low_corners, warped_low_masks)
            seam_masks = [
                seam_finder.resize(seam_mask, mask)
                for seam_mask, mask in zip(seam_masks, warped_final_masks)
            ]
        else:
            seam_masks = warped_final_masks

        # Exposure compensation
        if self.config.PERFORM_EXPOSURE_COMPENSATION:
            self.logger.info("Applying exposure compensation...")
            compensator = ExposureErrorCompensator()
            compensator.feed(low_corners, warped_low_imgs, warped_low_masks)
            compensated_imgs = [
                compensator.apply(idx, corner, img, mask)
                for idx, (img, mask, corner) in enumerate(
                    zip(warped_final_imgs, warped_final_masks, final_corners)
                )
            ]
        else:
            compensated_imgs = warped_final_imgs

        # Blender
        self.logger.info("Blending images...")
        blender = Blender()
        blender.prepare(final_corners, final_sizes)
        for img, mask, corner in zip(compensated_imgs, seam_masks, final_corners):
            blender.feed(img, mask, corner)
        panorama, _ = blender.blend()

        # Save panorama
        ts = datetime.now().timestamp()
        panorama_path = os.path.join(
            self.config.OUTPUT_DIR, f"pipeline_panorama_{ts}.png"
        )
        cv.imwrite(panorama_path, panorama)
        self.panorama_path = panorama_path

        self.logger.info(f"Panorama saved: {panorama_path}")
        self.logger.info("STEP 1 COMPLETE")

        return panorama_path

    def _save_full_calibration(self, cameras, warper_scale, image_sizes) -> Tuple[str, str]:
        """Save camera calibration parameters from step 1"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        camera_data = []
        for i, cam in enumerate(cameras):
            camera_data.append(
                {
                    "camera_index": i,
                    "focal": float(cam.focal),
                    "ppx": float(cam.ppx),
                    "ppy": float(cam.ppy),
                    "aspect": float(cam.aspect),
                    "R": cam.R.tolist(),
                    "t": cam.t.tolist(),
                }
            )

        calibration_data = {
            "timestamp": timestamp,
            "cameras": camera_data,
            "warper_scale": float(warper_scale),
            "image_sizes": {
                "original": list(image_sizes["original"]),
                "medium": list(image_sizes["medium"]),
                "low": list(image_sizes["low"]),
                "final": list(image_sizes["final"]),
            },
        }

        numpy_path = os.path.join(
            self.config.CALIBRATION_DIR, f"calibration_{timestamp}.npz"
        )
        json_path = os.path.join(self.config.CALIBRATION_DIR, f"calibration_{timestamp}.json")

        np.savez(
            numpy_path,
            cameras=camera_data,
            warper_scale=warper_scale,
            image_sizes=calibration_data["image_sizes"],
        )
        with open(json_path, "w") as f:
            json.dump(calibration_data, f, indent=2)

        return numpy_path, json_path

    ############################ STEP 2: MULTI-CAMERA CALIBRATION ############################
    def step2_extract_from_index(
        self, input_dir: str, index: int, panorama_path: Optional[str] = None
    ) -> str:
        """
        Extract N-camera calibration by selecting the `index`-th image from each ESP group.

        Args:
            input_dir: Directory with images containing esp_n in their names
            index: 0-based image index to pick from each ESP group
            panorama_path: reference panorama path; if None, use step1 output

        Returns:
            Path to saved multi calibration (.npz)
        """
        if panorama_path is None:
            panorama_path = self.panorama_path
        if panorama_path is None:
            raise ValueError("No panorama path provided and Step 1 hasn't been run")

        groups = self._group_images_by_esp(input_dir)
        image_paths = []
        for esp_name in self.config.ESP_CAMERAS:
            imgs = sorted(groups.get(esp_name, []))
            if len(imgs) <= index:
                raise ValueError(
                    f"{esp_name} has only {len(imgs)} images, cannot select index {index}"
                )
            image_paths.append(os.path.join(input_dir, imgs[index]))
            self.logger.info(f"{esp_name}: selected {imgs[index]} ({index+1}/{len(imgs)})")

        self.anchor_images = image_paths
        return self._extract_calibration(image_paths, panorama_path)

    def step2_extract_from_anchors(
        self, anchors: Union[List[str], Dict[str, str]], panorama_path: str
    ) -> str:
        """
        Extract N-camera calibration from explicit anchor images.

        Args:
            anchors: either a list of N paths ordered according to config.ESP_CAMERAS,
                     or a dict mapping 'esp_1'..'esp_N' to paths
            panorama_path: path to reference panorama

        Returns:
            Path to saved multi calibration (.npz)
        """
        if isinstance(anchors, dict):
            # Normalize keys like "1" or "esp_1" to esp_1 pattern
            normalized: Dict[str, str] = {}
            for key, val in anchors.items():
                if key.startswith("esp_"):
                    normalized[key] = val
                else:
                    normalized[f"esp_{key}"] = val
            image_paths = []
            for esp_name in self.config.ESP_CAMERAS:
                if esp_name not in normalized:
                    raise ValueError(f"Missing anchor for {esp_name}")
                image_paths.append(normalized[esp_name])
        else:
            # list assumed already in correct order
            if len(anchors) != len(self.config.ESP_CAMERAS):
                raise ValueError(
                    f"Expected {len(self.config.ESP_CAMERAS)} anchors, got {len(anchors)}"
                )
            image_paths = anchors

        self.anchor_images = image_paths
        return self._extract_calibration(image_paths, panorama_path)

    def _group_images_by_esp(self, input_dir: str) -> Dict[str, List[str]]:
        all_images = [
            f for f in sorted(os.listdir(input_dir)) if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        esp_groups: Dict[str, List[str]] = {esp: [] for esp in self.config.ESP_CAMERAS}
        for img in all_images:
            for esp in self.config.ESP_CAMERAS:
                if esp in img:
                    esp_groups[esp].append(img)
                    break
        return esp_groups

    def _extract_calibration(self, image_paths: List[str], panorama_path: str) -> str:
        """Extract N-camera calibration by matching anchors to the reference panorama."""
        self.logger.info("=" * 60)
        self.logger.info("STEP 2: EXTRACTING N-CAMERA CALIBRATION")
        self.logger.info("=" * 60)

        n = len(image_paths)
        self.logger.info(f"Using {n} anchor images:")
        for i, p in enumerate(image_paths):
            self.logger.info(f"  Cam {i+1}: {os.path.basename(p)}")

        self.logger.info(f"Reference panorama: {os.path.basename(panorama_path)}")
        panorama = cv.imread(panorama_path)
        if panorama is None:
            raise ValueError(f"Could not load panorama: {panorama_path}")

        # Prepare sizes
        images = Images.of(image_paths)
        medium_imgs = list(images.resize(Images.Resolution.MEDIUM))

        pano_h, pano_w = panorama.shape[:2]
        medium_h = images.get_image_size(medium_imgs[0])[1]
        scale = medium_h / pano_h
        pano_med_w = int(pano_w * scale)
        pano_med_h = int(pano_h * scale)
        panorama_medium = cv.resize(panorama, (pano_med_w, pano_med_h))

        self.logger.info(f"Panorama resized to: {(pano_med_h, pano_med_w)}")
        self.logger.info(f"Anchor medium sizes: {[img.shape[:2] for img in medium_imgs]}")

        # Feature detection across anchors + panorama
        self.logger.info("Detecting features...")
        finder = FeatureDetector(
            detector=self.config.DETECTOR_TYPE, nfeatures=self.config.N_FEATURES
        )
        all_images = medium_imgs + [panorama_medium]
        all_features = [finder.detect_features(img) for img in all_images]
        feat_counts = [len(f.getKeypoints()) for f in all_features]
        self.logger.info(f"Features - Anchors: {feat_counts[:n]}")
        self.logger.info(f"Features - Panorama: {feat_counts[n]}")

        # Match features
        self.logger.info("Matching features...")
        matcher = FeatureMatcher(matcher_type=self.config.MATCHER_TYPE)
        all_matches = matcher.match_features(all_features)
        conf_matrix = matcher.get_confidence_matrix(all_matches)
        self.logger.info("Confidence matrix:")
        self.logger.info(f"\n{conf_matrix}")

        # Estimate cameras (anchors + panorama)
        self.logger.info("Estimating camera parameters...")
        camera_estimator = CameraEstimator()
        camera_adjuster = CameraAdjuster(confidence_threshold=self.config.CONFIDENCE_THRESHOLD)
        wave_corrector = WaveCorrector()

        cameras = camera_estimator.estimate(all_features, all_matches)
        cameras = camera_adjuster.adjust(all_features, all_matches, cameras)
        cameras = wave_corrector.correct(cameras)

        # Keep first N cameras (for the N anchors in provided order)
        anchor_cameras = cameras[:n]

        # Warper scale for these N
        warper = Warper()
        warper.set_scale(anchor_cameras)

        image_sizes = {
            "original": images.sizes[0],
            "medium": images.get_image_size(medium_imgs[0]),
            "low": images.get_image_size(list(images.resize(Images.Resolution.LOW))[0]),
            "final": images.get_image_size(list(images.resize(Images.Resolution.FINAL))[0]),
            "panorama_original": (pano_w, pano_h),
            "panorama_medium": (pano_med_w, pano_med_h),
        }

        calib_path = self._save_multi_calibration(
            anchor_cameras, warper.scale, image_sizes, image_paths, panorama_path
        )
        self.calibration_file = calib_path

        self.logger.info("STEP 2 COMPLETE")
        return calib_path

    def _save_multi_calibration(
        self,
        cameras,
        warper_scale: float,
        image_sizes: dict,
        anchor_paths: List[str],
        panorama_path: str,
    ) -> str:
        """Save calibration for N cameras selected via anchors."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        camera_data = []
        for i, cam in enumerate(cameras):
            camera_data.append(
                {
                    "camera_index": i,
                    "source_image": os.path.basename(anchor_paths[i]),
                    "focal": float(cam.focal),
                    "ppx": float(cam.ppx),
                    "ppy": float(cam.ppy),
                    "aspect": float(cam.aspect),
                    "R": cam.R.tolist(),
                    "t": cam.t.tolist(),
                }
            )

        data = {
            "timestamp": timestamp,
            "method": "anchors_to_panorama_matching",
            "reference_panorama": os.path.basename(panorama_path),
            "anchors": [os.path.basename(p) for p in anchor_paths],
            "cameras": camera_data,
            "warper_scale": float(warper_scale),
            "image_sizes": {k: list(v) if isinstance(v, tuple) else v for k, v in image_sizes.items()},
        }

        npz_path = os.path.join(
            self.config.MULTI_CALIBRATION_DIR, f"multi_calibration_{timestamp}.npz"
        )
        json_path = os.path.join(
            self.config.MULTI_CALIBRATION_DIR, f"multi_calibration_{timestamp}.json"
        )

        np.savez(
            npz_path,
            cameras=camera_data,
            warper_scale=warper_scale,
            image_sizes=data["image_sizes"],
            metadata=data,
        )
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        self.logger.info("Multi calibration saved:")
        self.logger.info(f"  - NumPy: {npz_path}")
        self.logger.info(f"  - JSON: {json_path}")

        return npz_path

    ############################ STEP 3: N-CAMERA STITCHING ############################
    def step3_stitch_images(
        self,
        image_paths: List[str],
        calibration_file: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Stitch N new images using a saved calibration.

        Args:
            image_paths: List of N image paths (in the same order as calibration)
            calibration_file: Path to multi calibration (.npz/.json); uses last step2 if None
            output_path: Optional output path

        Returns:
            Path to stitched panorama
        """
        self.logger.info("=" * 60)
        self.logger.info("STEP 3: STITCHING N IMAGES")
        self.logger.info("=" * 60)

        if calibration_file is None:
            calibration_file = self.calibration_file
        if calibration_file is None:
            raise ValueError("No calibration file provided and Step 2 hasn't been run")

        n = len(image_paths)
        self.logger.info(f"Using calibration: {os.path.basename(calibration_file)}")
        for i, p in enumerate(image_paths):
            self.logger.info(f"  Img {i+1}: {os.path.basename(p)}")

        generator = MultiPanoramaGenerator(calibration_file, self.logger)

        if output_path is None:
            ts = datetime.now().timestamp()
            output_path = os.path.join(
                self.config.OUTPUT_DIR, f"multi_stitched_{ts}.png"
            )

        panorama = generator.create_panorama_from_images(
            image_paths,
            output_path,
            perform_exposure_compensation=self.config.PERFORM_EXPOSURE_COMPENSATION,
            perform_seam_finding=self.config.PERFORM_SEAM_FINDING,
        )

        self.logger.info(f"Panorama saved: {output_path}")
        self.logger.info("STEP 3 COMPLETE")
        return output_path

    ############################ FULL PIPELINE ############################
    def run_full_pipeline(self, input_dir: str, index: int) -> dict:
        """
        Run: Step1 (panorama) -> Step2 (N-calib via index) -> Step3 (stitch the same anchors)

        Args:
            input_dir: directory of frames
            index: index per ESP group for anchor selection

        Returns:
            Summary dict
        """
        self.logger.info("=" * 80)
        self.logger.info("STARTING FULL N-CAMERA PANORAMA PIPELINE")
        self.logger.info("=" * 80)

        start = datetime.now()
        try:
            pano_path = self.step1_generate_panorama(input_dir)
            calib_path = self.step2_extract_from_index(input_dir, index, pano_path)
            test_output = self.step3_stitch_images(self.anchor_images, calib_path)

            duration = datetime.now() - start
            results = {
                "panorama_path": pano_path,
                "calibration_file": calib_path,
                "test_stitch_path": test_output,
                "anchor_images": self.anchor_images,
                "duration": str(duration),
            }

            self.logger.info("=" * 80)
            self.logger.info("PIPELINE COMPLETE!")
            self.logger.info("=" * 80)
            self.logger.info(f"Duration: {duration}")
            self.logger.info(f"Panorama: {pano_path}")
            self.logger.info(f"Calibration: {calib_path}")
            self.logger.info(f"Test stitch: {test_output}")
            self.logger.info("=" * 80)
            return results
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise


############################ MULTI PANORAMA GENERATOR ############################
class MultiPanoramaGenerator:
    """Apply saved multi-camera calibration (N cameras) to stitch new images"""

    def __init__(self, calibration_file_path: str, logger=None):
        self.logger = logger or self._create_logger()
        self.load_calibration(calibration_file_path)

    def _create_logger(self):
        import logging

        return logging.getLogger("MultiPanoramaGenerator")

    def reconstruct_camera_params(self, camera_data: List[dict]):
        """Reconstruct cv2.detail.CameraParams list from saved dicts"""
        cameras = []
        for cam_data in camera_data:
            camera = cv.detail.CameraParams()
            camera.focal = cam_data["focal"]
            camera.ppx = cam_data["ppx"]
            camera.ppy = cam_data["ppy"]
            camera.aspect = cam_data["aspect"]
            camera.R = np.array(cam_data["R"], dtype=np.float32)
            camera.t = np.array(cam_data["t"], dtype=np.float32)
            cameras.append(camera)
        return cameras

    def load_calibration(self, calibration_file_path: str):
        """Load multi-camera calibration .npz or .json"""
        self.logger.info(
            f"Loading multi calibration: {os.path.basename(calibration_file_path)}"
        )

        if calibration_file_path.endswith(".npz"):
            data = np.load(calibration_file_path, allow_pickle=True)
            camera_data = data["cameras"]
            self.warper_scale = float(data["warper_scale"])
            # old numpy save could be dict -> .item()
            self.image_sizes = (
                data["image_sizes"].item()
                if hasattr(data["image_sizes"], "item")
                else data["image_sizes"].tolist()
            )
            self.metadata = (
                data["metadata"].item() if "metadata" in data else {"n": len(camera_data)}
            )
        elif calibration_file_path.endswith(".json"):
            with open(calibration_file_path, "r") as f:
                j = json.load(f)
            camera_data = j["cameras"]
            self.warper_scale = j["warper_scale"]
            self.image_sizes = j["image_sizes"]
            self.metadata = j
        else:
            raise ValueError("Calibration file must be .npz or .json")

        self.cameras = self.reconstruct_camera_params(camera_data)

        self.warper = Warper()
        self.warper.scale = self.warper_scale

        self.logger.info(
            f"Loaded calibration: {len(self.cameras)} cameras, scale: {self.warper_scale}"
        )

    def create_panorama_from_images(
        self,
        image_paths: List[str],
        output_path: Optional[str] = None,
        perform_exposure_compensation: bool = True,
        perform_seam_finding: bool = True,
    ):
        """Create panorama from N images using saved calibration (order must match calibration)"""
        n = len(image_paths)
        if n != len(self.cameras):
            self.logger.warning(
                f"Image count ({n}) != calibration camera count ({len(self.cameras)})"
            )

        self.logger.info("Processing images...")
        images = Images.of(image_paths)

        low_imgs = list(images.resize(Images.Resolution.LOW))
        final_imgs = list(images.resize(Images.Resolution.FINAL))

        # Verify expected size
        current_final_size = images.get_image_size(final_imgs[0])
        expected_size = tuple(self.image_sizes["final"])
        if current_final_size != expected_size:
            self.logger.warning(
                f"Image size mismatch! Expected: {expected_size}, Got: {current_final_size}"
            )

        # Warp low
        self.logger.info("Warping low resolution images...")
        low_sizes = images.get_scaled_img_sizes(Images.Resolution.LOW)
        aspect_low = images.get_ratio(Images.Resolution.MEDIUM, Images.Resolution.LOW)
        warped_low_imgs = list(self.warper.warp_images(low_imgs, self.cameras, aspect_low))
        warped_low_masks = list(
            self.warper.create_and_warp_masks(low_sizes, self.cameras, aspect_low)
        )
        low_corners, low_sizes = self.warper.warp_rois(low_sizes, self.cameras, aspect_low)

        # Warp final
        self.logger.info("Warping final resolution images...")
        final_sizes = images.get_scaled_img_sizes(Images.Resolution.FINAL)
        aspect_final = images.get_ratio(Images.Resolution.MEDIUM, Images.Resolution.FINAL)
        warped_final_imgs = list(self.warper.warp_images(final_imgs, self.cameras, aspect_final))
        warped_final_masks = list(
            self.warper.create_and_warp_masks(final_sizes, self.cameras, aspect_final)
        )
        final_corners, final_sizes = self.warper.warp_rois(
            final_sizes, self.cameras, aspect_final
        )

        # Seams
        if perform_seam_finding:
            self.logger.info("Finding optimal seams...")
            seam_finder = SeamFinder()
            seam_masks = seam_finder.find(warped_low_imgs, low_corners, warped_low_masks)
            seam_masks = [
                seam_finder.resize(seam_mask, mask)
                for seam_mask, mask in zip(seam_masks, warped_final_masks)
            ]
        else:
            seam_masks = warped_final_masks

        # Exposure compensation
        if perform_exposure_compensation:
            self.logger.info("Applying exposure compensation...")
            compensator = ExposureErrorCompensator()
            compensator.feed(low_corners, warped_low_imgs, warped_low_masks)
            compensated_imgs = [
                compensator.apply(idx, corner, img, mask)
                for idx, (img, mask, corner) in enumerate(
                    zip(warped_final_imgs, warped_final_masks, final_corners)
                )
            ]
        else:
            compensated_imgs = warped_final_imgs

        # Blend
        self.logger.info("Blending images...")
        blender = Blender()
        blender.prepare(final_corners, final_sizes)
        for img, mask, corner in zip(compensated_imgs, seam_masks, final_corners):
            blender.feed(img, mask, corner)
        panorama, _ = blender.blend()

        if output_path:
            cv.imwrite(output_path, panorama)
            self.logger.info(f"Panorama saved to: {output_path}")

        return panorama


############################ COMMAND LINE INTERFACE ############################
def main():
    parser = argparse.ArgumentParser(description="Complete Panorama Pipeline (N-camera)")
    parser.add_argument(
        "--mode", choices=["full", "stitch"], required=True, help="Pipeline mode"
    )

    # Full pipeline
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Input directory containing ESP camera images (required for full mode)",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=5,
        help="Index per ESP group to select as anchors for calibration (default: 5)",
    )

    # Stitch-only
    parser.add_argument(
        "--calibration_file",
        type=str,
        help="Path to saved multi calibration (.npz/.json) [stitch mode]",
    )
    parser.add_argument(
        "--images",
        nargs="+",
        help="N image paths to stitch (ordered to match calibration) [stitch mode]",
    )
    parser.add_argument("--output", type=str, help="Output path (optional)")

    # Optional config
    parser.add_argument(
        "--detector",
        choices=["orb", "sift", "brisk", "akaze"],
        default="orb",
        help="Feature detector type (default: orb)",
    )
    parser.add_argument(
        "--features", type=int, default=5000, help="Number of features (default: 5000)"
    )
    parser.add_argument(
        "--no-exposure-comp", action="store_true", help="Disable exposure compensation"
    )
    parser.add_argument(
        "--no-seam-finding", action="store_true", help="Disable seam finding"
    )

    args = parser.parse_args()

    if args.mode == "full" and not args.input_dir:
        parser.error("--input_dir is required for full mode")

    if args.mode == "stitch":
        if not args.calibration_file:
            parser.error("--calibration_file is required for stitch mode")
        if not args.images or len(args.images) < 3:
            parser.error("--images needs at least 3 image paths for stitching")

    # Configure
    config = PipelineConfig()
    config.DETECTOR_TYPE = args.detector
    config.N_FEATURES = args.features
    config.PERFORM_EXPOSURE_COMPENSATION = not args.no_exposure_comp
    config.PERFORM_SEAM_FINDING = not args.no_seam_finding

    pipeline = PanoramaPipeline(config)

    try:
        if args.mode == "full":
            results = pipeline.run_full_pipeline(args.input_dir, args.index)
            print("\n" + "=" * 60)
            print("PIPELINE RESULTS SUMMARY")
            print("=" * 60)
            print(f"Input Directory: {args.input_dir}")
            print(f"Index: {args.index}")
            print(f"Duration: {results['duration']}")
            print()
            print("Generated Files:")
            print(f"  Panorama: {results['panorama_path']}")
            print(f"  Calibration: {results['calibration_file']}")
            print(f"  Test Stitch: {results['test_stitch_path']}")
            print()
            print("Anchors Used:")
            for i, p in enumerate(results["anchor_images"]):
                print(f"  Cam {i+1}: {os.path.basename(p)}")
            print("=" * 60)

        elif args.mode == "stitch":
            output_path = pipeline.step3_stitch_images(
                args.images, args.calibration_file, args.output
            )
            print("\n" + "=" * 60)
            print("STITCHING RESULTS")
            print("=" * 60)
            print(f"Calibration File: {args.calibration_file}")
            print(f"Input Images ({len(args.images)}):")
            for i, p in enumerate(args.images):
                print(f"  Img {i+1}: {os.path.basename(p)}")
            print(f"Output: {output_path}")
            print("=" * 60)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


############################ UTILITIES ############################
def list_calibration_files(calibration_dir: Optional[str] = None):
    """List available calibration files"""
    if calibration_dir is None:
        calibration_dir = PipelineConfig.MULTI_CALIBRATION_DIR

    if not os.path.exists(calibration_dir):
        print(f"Calibration directory '{calibration_dir}' not found")
        return

    files = [f for f in os.listdir(calibration_dir) if f.endswith((".npz", ".json"))]
    if not files:
        print(f"No calibration files found in '{calibration_dir}'")
        return

    print(f"Available calibration files in '{calibration_dir}':")
    for f in sorted(files):
        print(f"  {f}")


def verify_images(image_paths: List[str]):
    """Verify images exist and are valid"""
    for i, p in enumerate(image_paths):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Image {i+1} not found: {p}")
        img = cv.imread(p)
        if img is None:
            raise ValueError(f"Invalid image file: {p}")
        print(f"Image {i+1}: {os.path.basename(p)} ({img.shape[1]}x{img.shape[0]})")


if __name__ == "__main__":
    # See CLI for usage
    main()
