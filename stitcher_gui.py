import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import math
import threading
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class SphericalPanoramaViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("360° Spherical Panorama Viewer")
        self.root.geometry("1400x900")
        
        # Panorama data
        self.panorama = None
        self.panorama_width = 2048
        self.panorama_height = 1024
        self.sphere_radius = 1.0
        
        # Camera controls
        self.yaw = 0.0
        self.pitch = 0.0
        self.fov = 90.0
        self.view_width = 800
        self.view_height = 600
        
        # Mouse interaction
        self.mouse_x = 0
        self.mouse_y = 0
        self.is_dragging = False
        
        # Feature matching parameters
        self.feature_threshold = 0.75
        self.min_match_count = 10
        self.blend_strength = 0.5
        self.ransac_threshold = 5.0
        
        # SIFT detector
        try:
            self.sift = cv2.SIFT_create()
        except AttributeError:
            # Fallback for older OpenCV versions
            self.sift = cv2.xfeatures2d.SIFT_create()
        
        self.setup_ui()
        self.initialize_panorama()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Feature matching controls
        params_frame = ttk.Frame(control_frame)
        params_frame.pack(fill=tk.X)
        
        # Row 1
        ttk.Label(params_frame, text="Feature Threshold:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.threshold_var = tk.DoubleVar(value=self.feature_threshold)
        threshold_scale = ttk.Scale(params_frame, from_=0.1, to=1.0, variable=self.threshold_var, 
                                  orient=tk.HORIZONTAL, length=150,
                                  command=self.update_threshold)
        threshold_scale.grid(row=0, column=1, padx=5)
        self.threshold_label = ttk.Label(params_frame, text=f"{self.feature_threshold:.2f}")
        self.threshold_label.grid(row=0, column=2, padx=(5, 20))
        
        ttk.Label(params_frame, text="Min Matches:").grid(row=0, column=3, sticky=tk.W, padx=(0, 5))
        self.min_matches_var = tk.IntVar(value=self.min_match_count)
        matches_spin = ttk.Spinbox(params_frame, from_=5, to=50, textvariable=self.min_matches_var, width=5)
        matches_spin.grid(row=0, column=4, padx=5)
        
        # Row 2
        ttk.Label(params_frame, text="Blend Strength:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5))
        self.blend_var = tk.DoubleVar(value=self.blend_strength)
        blend_scale = ttk.Scale(params_frame, from_=0.1, to=1.0, variable=self.blend_var, 
                               orient=tk.HORIZONTAL, length=150,
                               command=self.update_blend)
        blend_scale.grid(row=1, column=1, padx=5)
        self.blend_label = ttk.Label(params_frame, text=f"{self.blend_strength:.2f}")
        self.blend_label.grid(row=1, column=2, padx=(5, 20))
        
        ttk.Label(params_frame, text="RANSAC Threshold:").grid(row=1, column=3, sticky=tk.W, padx=(0, 5))
        self.ransac_var = tk.DoubleVar(value=self.ransac_threshold)
        ransac_spin = ttk.Spinbox(params_frame, from_=1.0, to=10.0, textvariable=self.ransac_var, 
                                 width=5, increment=0.5)
        ransac_spin.grid(row=1, column=4, padx=5)
        
        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(button_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Reset View", command=self.reset_view).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear Panorama", command=self.clear_panorama).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Panorama", command=self.save_panorama).pack(side=tk.LEFT, padx=5)
        
        # View controls
        view_frame = ttk.Frame(button_frame)
        view_frame.pack(side=tk.RIGHT)
        
        ttk.Label(view_frame, text="FOV:").pack(side=tk.LEFT, padx=5)
        self.fov_var = tk.DoubleVar(value=self.fov)
        fov_scale = ttk.Scale(view_frame, from_=30, to=150, variable=self.fov_var, 
                             orient=tk.HORIZONTAL, length=100,
                             command=self.update_fov)
        fov_scale.pack(side=tk.LEFT, padx=5)
        
        # Main content area
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas for panorama view
        self.canvas = tk.Canvas(content_frame, bg='black', width=self.view_width, height=self.view_height)
        self.canvas.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        # Store image reference to prevent garbage collection
        self.current_photo = None
        self.canvas_ready = False
        
        # Bind mouse events
        self.canvas.bind('<Button-1>', self.on_mouse_down)
        self.canvas.bind('<B1-Motion>', self.on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_mouse_up)
        self.canvas.bind('<MouseWheel>', self.on_mouse_wheel)
        self.canvas.bind('<Button-4>', self.on_mouse_wheel)  # Linux
        self.canvas.bind('<Button-5>', self.on_mouse_wheel)  # Linux
        self.canvas.bind('<Configure>', self.on_canvas_configure)
        
        # Enable drag and drop
        try:
            self.canvas.drop_target_register('DND_Files')
            self.canvas.dnd_bind('<<Drop>>', self.on_drop)
        except:
            # Fallback for systems without proper drag-drop support
            pass
        
        # Info panel
        info_frame = ttk.LabelFrame(content_frame, text="Panorama Info", padding=10)
        info_frame.pack(fill=tk.Y, side=tk.RIGHT, padx=(5, 0))
        
        self.info_text = tk.Text(info_frame, width=30, height=20, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=scrollbar.set)
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready - Drag images here or use Load Image button")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, pady=(5, 0))
        
    def update_threshold(self, value):
        self.feature_threshold = float(value)
        self.threshold_label.config(text=f"{self.feature_threshold:.2f}")
        
    def update_blend(self, value):
        self.blend_strength = float(value)
        self.blend_label.config(text=f"{self.blend_strength:.2f}")
        
    def update_fov(self, value):
        self.fov = float(value)
        self.redraw()
        
    def initialize_panorama(self):
        """Initialize with empty black panorama"""
        self.panorama = np.zeros((self.panorama_height, self.panorama_width, 3), dtype=np.uint8)
        self.has_content = False
        self.update_info("Panorama initialized\nSize: {}x{}\nImages: 0".format(
            self.panorama_width, self.panorama_height))
        
    def on_canvas_configure(self, event):
        """Handle canvas resize"""
        self.canvas_ready = True
        self.root.after_idle(self.redraw)
        
    def clear_panorama(self):
        """Clear the panorama"""
        self.initialize_panorama()
        self.status_var.set("Panorama cleared")
        self.redraw()
        
    def reset_view(self):
        """Reset camera view to center"""
        self.yaw = 0.0
        self.pitch = 0.0
        self.fov_var.set(90)
        self.fov = 90.0
        self.redraw()
        
    def load_image(self):
        """Load image from file dialog"""
        filename = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif")]
        )
        if filename:
            self.process_image(filename)
            
    def on_drop(self, event):
        """Handle drag and drop"""
        try:
            files = event.data.split()
            for file_path in files:
                file_path = file_path.strip('{}')  # Remove braces if present
                if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')):
                    self.process_image(file_path)
                    break
        except Exception as e:
            print(f"Drop error: {e}")
            
    def process_image(self, image_path):
        """Process and integrate new image into panorama"""
        self.status_var.set("Processing image...")
        self.root.update_idletasks()
        
        def process_thread():
            try:
                # Load image
                img = cv2.imread(image_path)
                if img is None:
                    self.root.after(0, lambda: messagebox.showerror("Error", "Could not load image"))
                    return
                    
                # Update parameters from UI
                self.feature_threshold = self.threshold_var.get()
                self.min_match_count = self.min_matches_var.get()
                self.blend_strength = self.blend_var.get()
                self.ransac_threshold = self.ransac_var.get()
                
                if not self.has_content:
                    # First image - place at center
                    self.place_first_image(img, os.path.basename(image_path))
                else:
                    # Subsequent images - feature match and stitch
                    self.stitch_image(img, os.path.basename(image_path))
                    
                self.root.after(0, self.redraw)
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Error processing image: {str(e)}"))
                self.root.after(0, lambda: self.status_var.set("Error processing image"))
                
        threading.Thread(target=process_thread, daemon=True).start()
        
    def place_first_image(self, img, filename):
        """Place first image at center of panorama"""
        # Resize image to fit panorama height while maintaining aspect ratio
        h, w = img.shape[:2]
        scale = min(self.panorama_height / h, self.panorama_width / (2 * w))
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized_img = cv2.resize(img, (new_w, new_h))
        
        # Place at center
        y_start = (self.panorama_height - new_h) // 2
        x_start = (self.panorama_width - new_w) // 2
        
        self.panorama[y_start:y_start+new_h, x_start:x_start+new_w] = resized_img
        self.has_content = True
        
        info_text = f"First image placed: {filename}\nPosition: center\nSize: {new_w}x{new_h}\nScale: {scale:.2f}"
        self.update_info(info_text)
        self.status_var.set("First image placed at center")
        
    def stitch_image(self, new_img, filename):
        """Stitch new image to existing panorama using feature matching"""
        # Convert to grayscale for feature detection
        panorama_gray = cv2.cvtColor(self.panorama, cv2.COLOR_BGR2GRAY)
        new_img_gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
        
        # Detect features
        kp1, des1 = self.sift.detectAndCompute(panorama_gray, None)
        kp2, des2 = self.sift.detectAndCompute(new_img_gray, None)
        
        if des1 is None or des2 is None:
            self.root.after(0, lambda: messagebox.showwarning("Warning", 
                f"No features detected in {filename}"))
            self.status_var.set(f"No features detected in {filename}")
            return
            
        # Match features using FLANN matcher for better performance
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        try:
            matches = flann.knnMatch(des1, des2, k=2)
        except cv2.error:
            # Fallback to BFMatcher if FLANN fails
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.feature_threshold * n.distance:
                    good_matches.append(m)
                    
        if len(good_matches) < self.min_match_count:
            self.root.after(0, lambda: messagebox.showwarning("Warning", 
                f"Not enough good matches found in {filename}: {len(good_matches)}/{self.min_match_count}"))
            self.status_var.set(f"Insufficient matches: {len(good_matches)}")
            return
            
        # Extract matched points
        src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography
        try:
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_threshold)
            if M is None:
                self.root.after(0, lambda: messagebox.showwarning("Warning", 
                    f"Could not compute homography for {filename}"))
                self.status_var.set("Homography computation failed")
                return
        except Exception as e:
            self.root.after(0, lambda: messagebox.showwarning("Warning", 
                f"Homography error for {filename}: {str(e)}"))
            self.status_var.set("Homography error")
            return
            
        # Warp new image
        warped_img = cv2.warpPerspective(new_img, M, (self.panorama_width, self.panorama_height))
        
        # Create blend mask
        blend_mask = np.zeros((self.panorama_height, self.panorama_width), dtype=np.float32)
        warped_gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
        blend_mask[warped_gray > 0] = 1.0
        
        # Apply Gaussian blur to mask for smooth blending
        blend_mask = cv2.GaussianBlur(blend_mask, (21, 21), 0)
        
        # Blend images
        for c in range(3):
            self.panorama[:, :, c] = (
                self.panorama[:, :, c] * (1 - blend_mask * self.blend_strength) +
                warped_img[:, :, c] * blend_mask * self.blend_strength
            ).astype(np.uint8)
            
        inliers = int(np.sum(mask)) if mask is not None else len(good_matches)
        info_text = (f"Image stitched: {filename}\n"
                    f"Features: {len(kp2)}\n"
                    f"Matches: {len(good_matches)}\n"
                    f"Inliers: {inliers}\n"
                    f"Blend strength: {self.blend_strength:.2f}")
        
        self.update_info(info_text)
        self.status_var.set(f"Image stitched - {len(good_matches)} matches, {inliers} inliers")
        
    def spherical_to_cartesian(self, theta, phi, r=1):
        """Convert spherical coordinates to cartesian"""
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.cos(phi)
        z = r * np.sin(phi) * np.sin(theta)
        return x, y, z
        
    def extract_view(self):
        """Extract current view from panorama based on yaw, pitch, and FOV"""
        if self.panorama is None:
            return np.zeros((self.view_height, self.view_width, 3), dtype=np.uint8)
            
        # Convert angles to radians
        yaw_rad = np.radians(self.yaw)
        pitch_rad = np.radians(self.pitch)
        fov_rad = np.radians(self.fov)
        
        # Create view coordinates
        view_img = np.zeros((self.view_height, self.view_width, 3), dtype=np.uint8)
        
        for y in range(self.view_height):
            for x in range(self.view_width):
                # Map pixel to view angles
                u = (x / self.view_width - 0.5) * fov_rad
                v = (0.5 - y / self.view_height) * fov_rad * (self.view_height / self.view_width)
                
                # Calculate direction vector
                dir_x = np.cos(v) * np.sin(u)
                dir_y = np.sin(v)
                dir_z = np.cos(v) * np.cos(u)
                
                # Rotate by yaw and pitch
                # Yaw rotation (around Y axis)
                temp_x = dir_x * np.cos(yaw_rad) + dir_z * np.sin(yaw_rad)
                temp_z = -dir_x * np.sin(yaw_rad) + dir_z * np.cos(yaw_rad)
                dir_x = temp_x
                dir_z = temp_z
                
                # Pitch rotation (around X axis)
                temp_y = dir_y * np.cos(pitch_rad) - dir_z * np.sin(pitch_rad)
                temp_z = dir_y * np.sin(pitch_rad) + dir_z * np.cos(pitch_rad)
                dir_y = temp_y
                dir_z = temp_z
                
                # Convert to spherical coordinates
                theta = np.arctan2(dir_x, dir_z)
                phi = np.arctan2(np.sqrt(dir_x**2 + dir_z**2), dir_y)
                
                # Map to panorama coordinates
                pano_x = int((theta / (2 * np.pi) + 0.5) * self.panorama_width) % self.panorama_width
                pano_y = int(phi / np.pi * self.panorama_height)
                pano_y = max(0, min(self.panorama_height - 1, pano_y))
                
                view_img[y, x] = self.panorama[pano_y, pano_x]
                
        return view_img
        
    def redraw(self):
        """Redraw the current view"""
        if not self.canvas_ready:
            return
            
        try:
            # Get canvas dimensions
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            # Skip if canvas not properly initialized
            if canvas_width <= 1 or canvas_height <= 1:
                self.root.after(100, self.redraw)
                return
            
            # Extract current view
            view_img = self.extract_view()
            
            # Convert BGR to RGB
            view_img_rgb = cv2.cvtColor(view_img, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_img = Image.fromarray(view_img_rgb)
            
            # Resize to canvas size
            pil_img = pil_img.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage and store reference
            self.current_photo = ImageTk.PhotoImage(pil_img)
            
            # Clear canvas and draw image
            self.canvas.delete("all")
            self.canvas.create_image(canvas_width//2, canvas_height//2, image=self.current_photo)
            
            # Draw crosshairs
            self.canvas.create_line(canvas_width//2 - 10, canvas_height//2, 
                                  canvas_width//2 + 10, canvas_height//2, 
                                  fill='red', width=2)
            self.canvas.create_line(canvas_width//2, canvas_height//2 - 10, 
                                  canvas_width//2, canvas_height//2 + 10, 
                                  fill='red', width=2)
            
            # Draw view info
            info_text = f"Yaw: {self.yaw:.1f}°  Pitch: {self.pitch:.1f}°  FOV: {self.fov:.1f}°"
            self.canvas.create_text(10, 10, text=info_text, anchor=tk.NW, fill='white', 
                                  font=('Arial', 10))
            
        except Exception as e:
            print(f"Redraw error: {e}")
            # Fallback: show error message on canvas
            self.canvas.delete("all")
            self.canvas.create_text(canvas_width//2, canvas_height//2, 
                                  text=f"Rendering error:\n{str(e)}", 
                                  fill='red', font=('Arial', 12), justify=tk.CENTER)
    
    def save_panorama(self):
        """Save current panorama to file"""
        if self.panorama is None or not self.has_content:
            messagebox.showwarning("Warning", "No panorama to save")
            return
            
        filename = filedialog.asksaveasfilename(
            title="Save Panorama",
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")]
        )
        
        if filename:
            # Convert BGR to RGB for saving
            panorama_rgb = cv2.cvtColor(self.panorama, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(panorama_rgb)
            pil_img.save(filename)
            self.status_var.set(f"Panorama saved to {filename}")
            
    def update_info(self, text):
        """Update info panel"""
        self.info_text.insert(tk.END, f"\n{'-'*30}\n{text}\n")
        self.info_text.see(tk.END)
        
    # Mouse event handlers
    def on_mouse_down(self, event):
        self.is_dragging = True
        self.mouse_x = event.x
        self.mouse_y = event.y
        
    def on_mouse_drag(self, event):
        if self.is_dragging:
            dx = event.x - self.mouse_x
            dy = event.y - self.mouse_y
            
            # Adjust sensitivity
            sensitivity = 0.5
            self.yaw -= dx * sensitivity
            self.pitch += dy * sensitivity
            
            # Clamp pitch
            self.pitch = max(-90, min(90, self.pitch))
            
            self.mouse_x = event.x
            self.mouse_y = event.y
            self.redraw()
            
    def on_mouse_up(self, event):
        self.is_dragging = False
        
    def on_mouse_wheel(self, event):
        # Zoom with mouse wheel
        if event.delta > 0 or event.num == 4:  # Zoom in
            self.fov = max(10, self.fov - 5)
        else:  # Zoom out
            self.fov = min(150, self.fov + 5)
        
        self.fov_var.set(self.fov)
        self.redraw()

def main():
    root = tk.Tk()
    
    # Try to enable drag and drop
    try:
        from tkinterdnd2 import TkinterDnD
        root.destroy()  # Destroy the regular Tk instance
        root = TkinterDnD.Tk()
        print("Drag-and-drop enabled")
    except ImportError:
        print("Warning: tkinterdnd2 not available, using file dialog only")
        
    app = SphericalPanoramaViewer(root)
    
    # Wait for UI to be ready, then initial draw
    def delayed_draw():
        app.canvas_ready = True
        app.redraw()
    
    root.after(500, delayed_draw)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        root.quit()

if __name__ == "__main__":
    main()