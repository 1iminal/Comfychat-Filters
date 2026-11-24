import torch
import numpy as np
import cv2

# MediaPipe FaceMesh (478-landmark) indices used to anchor the filter.
# 468/473 are the right/left iris centres, 1 is the nose tip.
_MP_LEFT_IRIS = 473
_MP_RIGHT_IRIS = 468
_MP_NOSE_TIP = 1

class ApplyComfychatFilterMHR:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mhr_pose_data": ("MHR_POSE_DATA", {"tooltip":"SAM3D Body prediction result"}),
                "background_image": ("IMAGE", {"tooltip":"Image to overlay the filter on"}), # Source frame [B, H, W, C]
                "filter_image": ("IMAGE",{"tooltip":"Filter image with alpha; Square dimensions recommended"}),     # Filter image with alpha channel [H, W, C]
                "offset_x": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01, "tooltip":"Horizontal filter offset"}),
                "offset_y": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01, "tooltip":"Vertical filter offset"}),
                "offset_z": ("FLOAT", {"default": 0.1, "min": -2.0, "max": 2.0, "step": 0.01, "tooltip":"Filter depth (3D perspective only)"}),
                "scale_x": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.01, "tooltip":"Filter scale along X axis"}),
                "scale_y": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.01, "tooltip":"Filter scale along Y axis"}),
                "tilt": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.5, "tooltip":"Filter tilt along its center (-180 to 180)"}),
                "rot_offset_x": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.5, "tooltip":"(3D perspective only) Filter rotation along X axis (-180 to 180); Rotations are applied in X-Y-Z(tilt) order."}),
                "rot_offset_y": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.5, "tooltip":"(3D perspective only) Filter rotation along Y axis (-180 to 180); Rotations are applied in X-Y-Z(tilt) order."}),
                "smoothing_window": ("INT", {"default": 3, "min": 1, "max": 21, "step": 2, "tooltip":"Number of sampled adjacent frames for motion smoothing (jitter reduction); 1 - no smoothing, 3-7 - optimal for video"}),
                "perspective": (["2D", "3D"], {"default": "2D", "tooltip":"2D - filter always faces the camera, 3D - makes filter follow head rotation"}),
                "occlusion_culling": ("BOOLEAN", {"default": True, "tooltip":"Hide filter when the face is turned away or occluded"}),
                "fade_threshold": ("FLOAT", {"default": 0.1, "min": 0.05, "max": 0.5, "step": 0.01, "tooltip":"Distance between eyes at which filter begins to fade out (relative to the total width of the image)"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("IMAGE", "FILTER_ONLY")
    FUNCTION = "apply_perspective_filter"
    CATEGORY = "MHR/Effects"
    SEARCH_ALIASES = ["snapchat", "face overlay"]

    def apply_perspective_filter(self, mhr_pose_data, background_image, filter_image, offset_x, offset_y, offset_z, scale_x, scale_y, tilt, rot_offset_x, rot_offset_y, smoothing_window, perspective, occlusion_culling, fade_threshold):
        bg_np = background_image.cpu().numpy()
        filter_np = filter_image[0].cpu().numpy()
        
        # If filter image lacks an alpha channel, add an empty opaque one
        if filter_np.shape[2] == 3:
            alpha = np.ones((filter_np.shape[0], filter_np.shape[1], 1), dtype=filter_np.dtype)
            filter_np = np.concatenate([filter_np, alpha], axis=2)
            
        f_h, f_w, *_ = filter_np.shape
        num_frames = len(bg_np)
        out_images = []
        out_filter_only = []
        
        if perspective == "2D":
            # Reference corners of the original filter image
            src_pts = np.array([
                [0, 0],         # Top left corner
                [f_w, 0],       # Top right corner
                [f_w, f_h],     # Bottom right corner
                [0, f_h]        # Bottom left corner
            ], dtype=np.float32)

            # --- STEP 0: APPLY TILT ---
            # Rotate the texture around its own center before projection
            if abs(tilt) > 0.001:
                rot_M = cv2.getRotationMatrix2D((f_w // 2, f_h // 2), tilt, 1.0)
                # Use BORDER_CONSTANT with a transparent background to prevent the edges from stretching
                filter_np = cv2.warpAffine(
                    (filter_np * 255).astype(np.uint8), 
                    rot_M, 
                    (f_w, f_h), 
                    flags=cv2.INTER_LINEAR, 
                    borderMode=cv2.BORDER_CONSTANT, 
                    borderValue=(0,0,0,0)
                ).astype(np.float32) / 255.0

            src_pts = np.array([[0, 0], [f_w, 0], [f_w, f_h], [0, f_h]], dtype=np.float32)

            # --- STEP 1: EXTRACTING RAW COORDINATES ---
            raw_left_eyes = []
            raw_right_eyes = []
            raw_tops = []

            for i in range(num_frames):
                frame_bg = bg_np[i]
                img_h, img_w, *_ = frame_bg.shape
                
                #frame_pose = mhr_pose_data[i] if isinstance(mhr_pose_data, list) else mhr_pose_data
                frame_pose = mhr_pose_data["frames"][i][0]
                
                # Extracting 2D projections of key points
                joints_2d = frame_pose.get("pred_keypoints_2d")
                
                if torch.is_tensor(joints_2d):
                    joints_2d = joints_2d.cpu().numpy()
                    
                # Function for scaling normalized coordinates to pixels
                def get_pix(idx):
                    pt = joints_2d[idx]
                    if np.max(pt) <= 1.0: # If the coordinates are from 0 to 1
                        return np.array([pt[0] * img_w, pt[1] * img_h])
                    return pt[:2]

                # Defining facial landmark points
                try:
                    raw_left_eyes.append(get_pix(1))   # Left eye
                    raw_right_eyes.append(get_pix(2))  # Right eye
                    raw_center = (raw_left_eyes[-1] + raw_right_eyes[-1]) / 2
                    eye_dist = np.linalg.norm(raw_right_eyes[-1] - raw_left_eyes[-1])
                    raw_tops.append(raw_center - np.array([0, eye_dist * 1.2])) 
                except IndexError:
                    fallback = raw_left_eyes[-1] if len(raw_left_eyes) > 0 else np.array([img_w/2, img_h/2])
                    raw_left_eyes.append(fallback)
                    raw_right_eyes.append(fallback)
                    raw_tops.append(fallback)

            raw_left_eyes = np.array(raw_left_eyes)
            raw_right_eyes = np.array(raw_right_eyes)
            raw_tops = np.array(raw_tops)

            # --- STEP 2: MOTION SMOOTHING ---
            smooth_left_eyes = np.copy(raw_left_eyes)
            smooth_right_eyes = np.copy(raw_right_eyes)
            smooth_tops = np.copy(raw_tops)

            if smoothing_window > 1 and num_frames > smoothing_window:
                half_w = smoothing_window // 2
                for i in range(num_frames):
                    start_idx = max(0, i - half_w)
                    end_idx = min(num_frames, i + half_w + 1)
                    
                    smooth_left_eyes[i] = np.mean(raw_left_eyes[start_idx:end_idx], axis=0)
                    smooth_right_eyes[i] = np.mean(raw_right_eyes[start_idx:end_idx], axis=0)
                    smooth_tops[i] = np.mean(raw_tops[start_idx:end_idx], axis=0)

            # --- STEP 3: RENDERING AND HOMOGRAPHY ---
            for i in range(num_frames):
                frame_bg = (bg_np[i] * 255).astype(np.uint8)
                img_h, img_w, *_ = frame_bg.shape
                visibility_multiplier = 1.0
                
                p_left = smooth_left_eyes[i]
                p_right = smooth_right_eyes[i]
                p_top = smooth_tops[i] # Forehead/Top of the head
                
                p_center = (p_left + p_right) / 2
                eye_dist = np.linalg.norm(p_right - p_left)
                
                # Calculate the normalized face size relative to the frame width
                normalized_face_size = eye_dist / img_w

                # Fade out logic
                if occlusion_culling:
                    # If the face is too small (turned in profile or facing away), we calculate the transparency
                    if normalized_face_size < fade_threshold:
                        # Smooth fade-out, becoming completely hidden upon reaching half the threshold
                        visibility_multiplier = max(0.0, (normalized_face_size - (fade_threshold / 2)) / (fade_threshold / 2))

                # If the filter is completely hidden, we return a clean background, saving resources
                if visibility_multiplier <= 0.0:
                    out_images.append(bg_np[i])
                    out_filter_only.append(np.zeros((img_h, img_w, 4), dtype=bg_np[i].dtype))
                    continue

                # Filter center and scaling adjustment
                face_width = eye_dist * 2.0 * scale_x
                face_height = eye_dist * 2.0 * scale_y
                
                # Face orientation vectors (define pitch and yaw)
                vector_x = (p_right - p_left) / np.max([eye_dist, 0.001])
                vector_y = -np.array([-vector_x[1], vector_x[0]])  # Perpendicular for the vertical axis
                
                # Vertical & horizontal shift (offset_y allows you to move the mask up onto the forehead/hat or down onto the mouth, offset_x - to move it let/right)
                center = p_center + (vector_x * (offset_x * face_width)) + (vector_y * (offset_y * face_height))

                # --- CALCULATING 4 POTENTIAL DESTINATION POINTS ---
                # We construct a 3D-oriented quadrilateral directly on the character's face
                dst_top_left = center - (vector_x * (face_width / 2)) - (vector_y * (face_height / 2))
                dst_top_right = center + (vector_x * (face_width / 2)) - (vector_y * (face_height / 2))
                dst_bottom_right = center + (vector_x * (face_width / 2)) + (vector_y * (face_height / 2))
                dst_bottom_left = center - (vector_x * (face_width / 2)) + (vector_y * (face_height / 2))
                
                dst_pts = np.array([dst_top_right, dst_top_left, dst_bottom_left, dst_bottom_right], dtype=np.float32)
                    
                # --- HOMOGRAPHY (Key step for Pitch/Yaw/Roll) ---
                # Calculate a 3x3 transformation matrix that compresses and stretches the texture
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                
                # Project the filter image onto the background frame, accounting for perspective
                fg_warped = cv2.warpPerspective(
                    (filter_np * 255).astype(np.uint8), 
                    M, 
                    (img_w, img_h), 
                    flags=cv2.INTER_LINEAR, 
                    borderMode=cv2.BORDER_CONSTANT, 
                    borderValue=(0,0,0,0)
                )

                # --- ALPHA BLENDING ---
                # Extract the transparency mask from the transformed filter and apply visibility_multiplier
                alpha_mask = (fg_warped[:, :, 3:4] / 255.0) * visibility_multiplier
                fg_rgb = fg_warped[:, :, 0:3]
                
                # Overlay onto the original frame
                frame_out = (fg_rgb * alpha_mask + frame_bg * (1.0 - alpha_mask)).astype(np.uint8)
                out_images.append(frame_out.astype(np.float32) / 255.0)
                # Also save filter without the background just in case
                out_filter_only.append((fg_warped * visibility_multiplier).astype(np.float32) / 255.0)

        else:
            # --- STEP 1: EXTRACTION AND SMOOTHING OF 3D COORDINATES ---
            # We need 3 reference points in 3D to establish a rigid orientation of the face
            raw_3d_nose = []
            raw_3d_left = []
            raw_3d_right = []
            raw_2d_center = [] # For tracking size on screen (hiding the mask)

            for i in range(num_frames):
                frame_bg = bg_np[i]
                img_h, img_w, *_ = frame_bg.shape
                
                frame_pose = mhr_pose_data["frames"][i][0]
                
                # Extract the original array of 3D keypoints
                joints_3d = frame_pose.get("pred_keypoints_3d")
                cam_t = frame_pose.get("pred_cam_t")
                joints_2d = frame_pose.get("pred_keypoints_2d")
                
                # Also extract rotation matrices for accurate face plane calculations
                joint_rots = frame_pose.get("pred_global_rots")
                
                if torch.is_tensor(joints_3d): joints_3d = joints_3d.cpu().numpy()
                if torch.is_tensor(cam_t): cam_t = cam_t.cpu().numpy()
                if torch.is_tensor(joints_2d): joints_2d = joints_2d.cpu().numpy()
                if torch.is_tensor(joint_rots): joint_rots = joint_rots.cpu().numpy()

                try:
                    # Take the 3D coordinates (X, Y, Z)
                    raw_3d_nose.append(joints_3d[0][:3] + cam_t)
                    raw_3d_left.append(joints_3d[1][:3] + cam_t)  # Left eye
                    raw_3d_right.append(joints_3d[2][:3] + cam_t) # Right eye
                    
                    # Additionally, we take a 2D projection to calculate the fade effect.
                    pt_l = joints_2d[1][:2] # Left eye
                    pt_r = joints_2d[2][:2] # Right eye
                    if np.max(pt_l) <= 1.0:
                        raw_2d_center.append(np.linalg.norm(pt_r - pt_l) * img_w)
                    else:
                        raw_2d_center.append(np.linalg.norm(pt_r - pt_l))
                except (IndexError, TypeError):
                    fallback = raw_3d_left[-1] if len(raw_3d_left) > 0 else np.array([0, 0, 0])
                    raw_3d_nose.append(fallback)
                    raw_3d_left.append(fallback)
                    raw_3d_right.append(fallback)
                    raw_2d_center.append(img_w * 0.15)

            # Apply moving average to 3D arrays
            smooth_3d_nose = np.copy(raw_3d_nose)
            smooth_3d_left = np.copy(raw_3d_left)
            smooth_3d_right = np.copy(raw_3d_right)

            if smoothing_window > 1 and num_frames > smoothing_window:
                half_w = smoothing_window // 2
                for i in range(num_frames):
                    s, e = max(0, i - half_w), min(num_frames, i + half_w + 1)
                    smooth_3d_nose[i] = np.mean(raw_3d_nose[s:e], axis=0)
                    smooth_3d_left[i] = np.mean(raw_3d_left[s:e], axis=0)
                    smooth_3d_right[i] = np.mean(raw_3d_right[s:e], axis=0)

            # --- STEP 2: FRAME-BY-FRAME 3D RENDERING VIA A PERSPECTIVE CAMERA ---
            for i in range(num_frames):
                frame_bg = (bg_np[i] * 255).astype(np.uint8)
                img_h, img_w, _ = frame_bg.shape
                
                # Logic for auto-hiding a mask based on a 2D projection
                if occlusion_culling:
                    normalized_size = raw_2d_center[i] / img_w
                    if normalized_size < fade_threshold:
                        visibility = max(0.0, (normalized_size - (fade_threshold / 2)) / (fade_threshold / 2))
                        if visibility <= 0.0:
                            out_images.append(bg_np[i])
                            continue
                    else: visibility = 1.0
                else: visibility = 1.0

                # Get the current stable 3D points
                p3d_nose = smooth_3d_nose[i]
                p3d_left = smooth_3d_left[i]
                p3d_right = smooth_3d_right[i]
                
                p3d_center = (p3d_left + p3d_right) / 2.0
                eye_dist_3d = np.linalg.norm(p3d_right - p3d_left)
                
                # Build more face plane points
                #head_rot_matrix = joint_rots[16]
                #local_vec_up = head_rot_matrix[:, 0]
                #local_vec_up /= np.max([np.linalg.norm(local_vec_up), 0.001])
                #p3d_top = p3d_center + (local_vec_up * eye_dist_3d)
                #p3d_bottom = p3d_center - (local_vec_up * eye_dist_3d)

                # --- MATHEMATICAL CONSTRUCTION OF THE FACE BASIS (AXES) ---
                # X-axis: from the left eye to the right eye
                axis_x = (p3d_right - p3d_left) / np.max([eye_dist_3d, 0.001])
                
                # Vector from the nose toward the center of the eyes (an "upward" direction along the face, though not a perfect perpendicular)
                #vec_up = p3d_center - p3d_nose
                vec_up = p3d_center - p3d_nose
                
                # Z-axis: cross product (direction pointing strictly "forward" from the face)
                axis_z = np.cross(axis_x, vec_up)
                axis_z /= np.max([np.linalg.norm(axis_z), 0.001])
                
                # Y-axis: strictly perpendicular to X and Z (the "downward" direction along the face)
                axis_y = np.cross(axis_z, axis_x)
                axis_y /= np.max([np.linalg.norm(axis_y), 0.001])
                
                # Take axis vectors from head rotation matrix
                #head_rot_matrix = joint_rots[16]
                #axis_x = head_rot_matrix[:, 0]
                #axis_y = head_rot_matrix[:, 1]
                #axis_z = head_rot_matrix[:, 2]

                # Optional texture rotation (tilt) along the Z-axis
                if abs(tilt) > 0.001 or abs(rot_offset_x) > 0.001 or abs(rot_offset_y) > 0.001:
                    rad_x = np.radians(rot_offset_x)
                    rad_y = np.radians(rot_offset_y)
                    rad_z = np.radians(tilt)
                    
                    rot_x = np.array([
                        [1,    0,             0           ],
                        [0,    np.cos(rad_x), -np.sin(rad_x)],
                        [0,    np.sin(rad_x),  np.cos(rad_x)]
                    ])
                    rot_y = np.array([
                        [np.cos(rad_y),  0, np.sin(rad_y)],
                        [0,              1,             0],
                        [-np.sin(rad_y), 0, np.cos(rad_y)]
                    ])
                    rot_z = np.array([
                        [np.cos(rad_z), -np.sin(rad_z), 0],
                        [np.sin(rad_z),  np.cos(rad_z), 0],
                        [0,              0,             1]
                    ])
                    
                    # X → Y → Z  (column-vector convention)
                    rot = rot_z @ rot_y @ rot_x
                    
                    basis = np.stack([axis_x, axis_y, axis_z], axis=1)
                    basis = np.dot(basis, rot)
                    axis_x, axis_y, axis_z = basis[:, 0], basis[:, 1], basis[:, 2]

                # Adjusting the physical size of a virtual 3D plan to match facial anthropometry
                w_3d = eye_dist_3d * 2.2 * scale_x
                h_3d = eye_dist_3d * 2.2 * scale_y

                # Apply user offsets based on the face's local 3D axes (left-right, up-down, depth/closer-further)
                p3d_center += axis_x * (offset_x * w_3d) + axis_y * (offset_y * h_3d) + axis_z * (offset_z * eye_dist_3d)

                # Constructing the coordinates of the filter's 4 corners in a TRUE 3D space around the center of the face
                obj_top_left     = p3d_center + (axis_x * (w_3d / 2.0)) + (axis_y * (h_3d / 2.0))
                obj_top_right    = p3d_center - (axis_x * (w_3d / 2.0)) + (axis_y * (h_3d / 2.0))
                obj_bottom_right = p3d_center - (axis_x * (w_3d / 2.0)) - (axis_y * (h_3d / 2.0))
                obj_bottom_left  = p3d_center + (axis_x * (w_3d / 2.0)) - (axis_y * (h_3d / 2.0))
                
                object_points_3d = np.array([obj_top_left, obj_top_right, obj_bottom_right, obj_bottom_left], dtype=np.float32)

                # We need the corresponding 2D pixels on the screen. 
                # Since MHR provides 3D coordinates already in camera space,
                # we can use a simplified orthographic or weak-perspective projection. 
                # We project the 3D points onto a 2D screen based on MHR’s rendering parameters. 
                # Scale and focal length are encoded within the axis ratios used by MHR. 
                # Most MHR implementations employ a weak-perspective model, where Z represents depth (distance to the camera).
                
                dst_pts = []
                for pt in object_points_3d:
                    # Projection: divide X and Y by Z (distance to the camera), simulating a real-world lens.
                    # If your MHR pipeline already has the joints scaled specifically for the screen, division by Z is not required. 
                    # However, for an accurate "yaw" effect, dividing by the depth Z creates the necessary asymmetry between the sides!
                    z_coord = np.max([pt[2], 0.1])
                    
                    # Projection factor (the MHR focal length is typically normalized to around 1.0–2.0)
                    # Convert metric coordinates into screen pixels relative to the center of the frame
                    screen_x = (pt[0] / z_coord) * img_w + (img_w / 2.0)
                    screen_y = (pt[1] / z_coord) * img_h + (img_h / 2.0)
                    
                    dst_pts.append([screen_x, screen_y])
                    
                dst_pts = np.array(dst_pts, dtype=np.float32)

                # Original 2D coordinates of the texture corners
                src_pts = np.array([[0, 0], [f_w, 0], [f_w, f_h], [0, f_h]], dtype=np.float32)

                # Homography based on 3D projection
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                
                fg_warped = cv2.warpPerspective(
                    (filter_np * 255).astype(np.uint8), M, (img_w, img_h), 
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0)
                )

                # Alpha blending
                alpha_mask = (fg_warped[:, :, 3:4] / 255.0) * visibility
                fg_rgb = fg_warped[:, :, 0:3]
                frame_out = (fg_rgb * alpha_mask + frame_bg * (1.0 - alpha_mask)).astype(np.uint8)
                out_images.append(frame_out.astype(np.float32) / 255.0)
                # Also save filter without the background just in case
                out_filter_only.append((fg_warped * visibility).astype(np.float32) / 255.0)

        return (torch.from_numpy(np.array(out_images)), torch.from_numpy(np.array(out_filter_only)))

class ApplyComfychatFilterMediaPipe:
    """Snapchat-style filter overlay driven by MediaPipe `FACE_LANDMARKS`.

    Consumes the output of the built-in `Detect Face Landmarks (MediaPipe)`
    node (the `FACE_LANDMARKS` type) instead of SAM3D/MHR pose data. The 2D
    (billboard) mode anchors the filter on the two iris centres; the 3D mode
    lifts the landmarks into camera space with the per-face transformation
    matrix and projects the filter plane with recovered pinhole intrinsics.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "face_landmarks": ("FACE_LANDMARKS", {"tooltip": "Face landmarks from Detect Face Landmarks (MediaPipe)"}),
                "background_image": ("IMAGE", {"tooltip": "Image to overlay the filter on"}),
                "filter_image": ("IMAGE", {"tooltip": "Filter image with alpha; Square dimensions recommended"}),
                "face_index": ("INT", {"default": 0, "min": 0, "max": 15, "step": 1, "tooltip": "Which detected face to use (0 = first)"}),
                "offset_x": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01, "tooltip": "Horizontal filter offset"}),
                "offset_y": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01, "tooltip": "Vertical filter offset"}),
                "offset_z": ("FLOAT", {"default": 0.1, "min": -2.0, "max": 2.0, "step": 0.01, "tooltip": "Filter depth (3D perspective only)"}),
                "scale_x": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.01, "tooltip": "Filter scale along X axis"}),
                "scale_y": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.01, "tooltip": "Filter scale along Y axis"}),
                "tilt": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.5, "tooltip": "Filter tilt along its center (-180 to 180)"}),
                "rot_offset_x": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.5, "tooltip": "(3D perspective only) Filter rotation along X axis (-180 to 180); Rotations are applied in X-Y-Z(tilt) order."}),
                "rot_offset_y": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.5, "tooltip": "(3D perspective only) Filter rotation along Y axis (-180 to 180); Rotations are applied in X-Y-Z(tilt) order."}),
                "smoothing_window": ("INT", {"default": 3, "min": 1, "max": 21, "step": 2, "tooltip": "Number of sampled adjacent frames for motion smoothing (jitter reduction); 1 - no smoothing, 3-7 - optimal for video"}),
                "perspective": (["2D", "3D"], {"default": "2D", "tooltip": "2D - filter always faces the camera, 3D - makes filter follow head rotation"}),
                "occlusion_culling": ("BOOLEAN", {"default": True, "tooltip": "Hide filter when the face is turned away or occluded"}),
                "fade_threshold": ("FLOAT", {"default": 0.1, "min": 0.05, "max": 0.5, "step": 0.01, "tooltip": "Distance between eyes at which filter begins to fade out (relative to the total width of the image)"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("IMAGE", "FILTER_ONLY")
    FUNCTION = "apply_perspective_filter"
    CATEGORY = "MediaPipe/Effects"
    SEARCH_ALIASES = ["snapchat", "face overlay", "mediapipe"]

    @staticmethod
    def _estimate_intrinsics(p_cam, l2):
        """Recover pinhole intrinsics (fx, fy, cx, cy) from camera-space 3D
        points and their 2D pixel projections via least squares.

        `p_cam` is (N, 3) in camera space, `l2` is (N, 2) in pixels. Returns
        `(fx, fy, cx, cy)` or `None` when the fit is degenerate.
        """
        z = p_cam[:, 2]
        mask = np.abs(z) > 1e-6
        if int(mask.sum()) < 16:
            return None
        a = p_cam[mask, 0] / z[mask]
        b = p_cam[mask, 1] / z[mask]
        u = l2[mask, 0]
        v = l2[mask, 1]
        ones = np.ones_like(a)
        # u = fx * a + cx ;  v = fy * b + cy
        sol_u, *_ = np.linalg.lstsq(np.stack([a, ones], axis=1), u, rcond=None)
        sol_v, *_ = np.linalg.lstsq(np.stack([b, ones], axis=1), v, rcond=None)
        fx, cx = float(sol_u[0]), float(sol_u[1])
        fy, cy = float(sol_v[0]), float(sol_v[1])
        # MediaPipe's camera space has Z<0 in front of the lens, so the recovered
        # focal lengths can carry a sign (fx<0, fy>0 for the default convention).
        # Only reject when the magnitude is degenerate.
        if abs(fx) <= 1.0 or abs(fy) <= 1.0:
            return None
        return (fx, fy, cx, cy)

    @staticmethod
    def _project_2d(p_left, p_right, offset_x, offset_y, scale_x, scale_y, extra_tilt):
        """Billboard placement from two 2D eye centres. Returns dst_pts in the
        [TR, TL, BL, BR] order expected by the shared homography, or None."""
        eye_dist = float(np.linalg.norm(p_right - p_left))
        if eye_dist < 1e-3:
            return None
        p_center = (p_left + p_right) / 2.0
        vector_x = (p_right - p_left) / eye_dist
        vector_y = -np.array([-vector_x[1], vector_x[0]])
        face_width = eye_dist * 2.0 * scale_x
        face_height = eye_dist * 2.0 * scale_y
        center = p_center + (vector_x * (offset_x * face_width)) + (vector_y * (offset_y * face_height))
        hw, hh = face_width / 2.0, face_height / 2.0
        corners = np.array([
            center - vector_x * hw - vector_y * hh,  # top-left
            center + vector_x * hw - vector_y * hh,  # top-right
            center + vector_x * hw + vector_y * hh,  # bottom-right
            center - vector_x * hw + vector_y * hh,  # bottom-left
        ])
        if abs(extra_tilt) > 0.001:
            th = np.radians(extra_tilt)
            rot = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
            corners = (corners - center) @ rot.T + center
        # Match the MHR node's src/dst correspondence: [TR, TL, BL, BR]
        return np.array([corners[1], corners[0], corners[3], corners[2]], dtype=np.float32)

    @staticmethod
    def _project_3d(p3d_left, p3d_right, p3d_nose, intrinsics, offset_x, offset_y, offset_z,
                    scale_x, scale_y, tilt, rot_offset_x, rot_offset_y):
        """Place the filter on the 3D face plane and project to screen pixels.

        `p3d_*` are camera-space points, `intrinsics` is (fx, fy, cx, cy).
        Returns dst_pts in the [TL, TR, BR, BL] order, or None if degenerate.
        """
        fx, fy, cx, cy = intrinsics
        p3d_center = (p3d_left + p3d_right) / 2.0
        eye_dist_3d = float(np.linalg.norm(p3d_right - p3d_left))
        if eye_dist_3d < 1e-6:
            return None

        # Face basis (axes) in camera space
        axis_x = (p3d_right - p3d_left) / max(eye_dist_3d, 0.001)
        vec_up = p3d_center - p3d_nose
        axis_z = np.cross(axis_x, vec_up)
        axis_z /= max(np.linalg.norm(axis_z), 0.001)
        axis_y = np.cross(axis_z, axis_x)
        axis_y /= max(np.linalg.norm(axis_y), 0.001)

        # Optional user rotations, applied in X -> Y -> Z(tilt) order
        if abs(tilt) > 0.001 or abs(rot_offset_x) > 0.001 or abs(rot_offset_y) > 0.001:
            rad_x = np.radians(rot_offset_x)
            rad_y = np.radians(rot_offset_y)
            rad_z = np.radians(tilt)
            rot_x = np.array([[1, 0, 0], [0, np.cos(rad_x), -np.sin(rad_x)], [0, np.sin(rad_x), np.cos(rad_x)]])
            rot_y = np.array([[np.cos(rad_y), 0, np.sin(rad_y)], [0, 1, 0], [-np.sin(rad_y), 0, np.cos(rad_y)]])
            rot_z = np.array([[np.cos(rad_z), -np.sin(rad_z), 0], [np.sin(rad_z), np.cos(rad_z), 0], [0, 0, 1]])
            rot = rot_z @ rot_y @ rot_x
            basis = np.stack([axis_x, axis_y, axis_z], axis=1)
            basis = np.dot(basis, rot)
            axis_x, axis_y, axis_z = basis[:, 0], basis[:, 1], basis[:, 2]

        # Physical size of the virtual plane + user offsets in the face frame
        w_3d = eye_dist_3d * 2.2 * scale_x
        h_3d = eye_dist_3d * 2.2 * scale_y
        p3d_center = p3d_center + axis_x * (offset_x * w_3d) + axis_y * (offset_y * h_3d) + axis_z * (offset_z * eye_dist_3d)

        obj_top_left = p3d_center + axis_x * (w_3d / 2.0) + axis_y * (h_3d / 2.0)
        obj_top_right = p3d_center - axis_x * (w_3d / 2.0) + axis_y * (h_3d / 2.0)
        obj_bottom_right = p3d_center - axis_x * (w_3d / 2.0) - axis_y * (h_3d / 2.0)
        obj_bottom_left = p3d_center + axis_x * (w_3d / 2.0) - axis_y * (h_3d / 2.0)
        object_points_3d = np.array([obj_top_left, obj_top_right, obj_bottom_right, obj_bottom_left])

        # Pinhole projection with the recovered intrinsics
        dst_pts = []
        for pt in object_points_3d:
            z = pt[2]
            if abs(z) < 1e-6:
                return None
            dst_pts.append([fx * (pt[0] / z) + cx, fy * (pt[1] / z) + cy])
        return np.array(dst_pts, dtype=np.float32)

    def apply_perspective_filter(self, face_landmarks, background_image, filter_image, face_index,
                                 offset_x, offset_y, offset_z, scale_x, scale_y, tilt,
                                 rot_offset_x, rot_offset_y, smoothing_window, perspective,
                                 occlusion_culling, fade_threshold):
        bg_np = background_image.cpu().numpy()
        filter_np = filter_image[0].cpu().numpy()

        # If the filter image lacks an alpha channel, add an opaque one
        if filter_np.shape[2] == 3:
            alpha = np.ones((filter_np.shape[0], filter_np.shape[1], 1), dtype=filter_np.dtype)
            filter_np = np.concatenate([filter_np, alpha], axis=2)

        f_h, f_w, *_ = filter_np.shape
        num_frames = len(bg_np)
        frames = face_landmarks["frames"]

        # In 2D (billboard) mode the tilt is baked into the texture once, exactly
        # like the MHR node. In 3D mode it is applied per-frame as a Z rotation.
        if perspective == "2D" and abs(tilt) > 0.001:
            rot_M = cv2.getRotationMatrix2D((f_w // 2, f_h // 2), tilt, 1.0)
            filter_np = cv2.warpAffine(
                (filter_np * 255).astype(np.uint8), rot_M, (f_w, f_h),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
            ).astype(np.float32) / 255.0

        # --- STEP 1: EXTRACT RAW FACE ANCHORS ---
        raw_left_eyes = []
        raw_right_eyes = []
        raw_3d_left = []
        raw_3d_right = []
        raw_3d_nose = []
        raw_intrinsics = []
        valid_3d = []

        for i in range(num_frames):
            img_h, img_w, *_ = bg_np[i].shape
            faces = frames[i] if i < len(frames) else []
            face = faces[face_index] if face_index < len(faces) else None

            if face is None:
                raw_left_eyes.append(np.array([img_w / 2, img_h / 2]))
                raw_right_eyes.append(np.array([img_w / 2, img_h / 2]))
                raw_3d_left.append(np.array([0.0, 0.0, 0.0]))
                raw_3d_right.append(np.array([0.0, 0.0, 0.0]))
                raw_3d_nose.append(np.array([0.0, 0.0, 0.0]))
                raw_intrinsics.append(None)
                valid_3d.append(False)
                continue

            l2 = np.asarray(face["landmarks_xy"], dtype=np.float64)
            l3 = np.asarray(face["landmarks_3d"], dtype=np.float64)
            m = face.get("transformation_matrix")

            eye_l = l2[_MP_LEFT_IRIS]
            eye_r = l2[_MP_RIGHT_IRIS]
            raw_left_eyes.append(eye_l.copy())
            raw_right_eyes.append(eye_r.copy())

            ok = False
            if m is not None:
                m = np.asarray(m, dtype=np.float64)
                if m.shape == (4, 4):
                    p_cam = l3 @ m[:3, :3].T + m[:3, 3]
                    intr = self._estimate_intrinsics(p_cam, l2)
                    if intr is not None:
                        raw_3d_left.append(p_cam[_MP_LEFT_IRIS].copy())
                        raw_3d_right.append(p_cam[_MP_RIGHT_IRIS].copy())
                        raw_3d_nose.append(p_cam[_MP_NOSE_TIP].copy())
                        raw_intrinsics.append(intr)
                        valid_3d.append(True)
                        ok = True

            if not ok:
                raw_3d_left.append(np.array([0.0, 0.0, 0.0]))
                raw_3d_right.append(np.array([0.0, 0.0, 0.0]))
                raw_3d_nose.append(np.array([0.0, 0.0, 0.0]))
                raw_intrinsics.append(None)
                valid_3d.append(False)

        raw_left_eyes = np.array(raw_left_eyes)
        raw_right_eyes = np.array(raw_right_eyes)
        raw_3d_left = np.array(raw_3d_left)
        raw_3d_right = np.array(raw_3d_right)
        raw_3d_nose = np.array(raw_3d_nose)

        # --- STEP 2: MOTION SMOOTHING ---
        smooth_left = np.copy(raw_left_eyes)
        smooth_right = np.copy(raw_right_eyes)
        smooth_3d_left = np.copy(raw_3d_left)
        smooth_3d_right = np.copy(raw_3d_right)
        smooth_3d_nose = np.copy(raw_3d_nose)

        if smoothing_window > 1 and num_frames > smoothing_window:
            half_w = smoothing_window // 2
            for i in range(num_frames):
                s = max(0, i - half_w)
                e = min(num_frames, i + half_w + 1)
                smooth_left[i] = np.mean(raw_left_eyes[s:e], axis=0)
                smooth_right[i] = np.mean(raw_right_eyes[s:e], axis=0)
                idx = [j for j in range(s, e) if valid_3d[j]]
                if idx:
                    smooth_3d_left[i] = np.mean(raw_3d_left[idx], axis=0)
                    smooth_3d_right[i] = np.mean(raw_3d_right[idx], axis=0)
                    smooth_3d_nose[i] = np.mean(raw_3d_nose[idx], axis=0)
                else:
                    valid_3d[i] = False  # no 3D support in window -> 2D fallback

        # --- STEP 3: RENDER ---
        out_images = []
        out_filter_only = []

        for i in range(num_frames):
            frame_bg = (bg_np[i] * 255).astype(np.uint8)
            img_h, img_w, *_ = frame_bg.shape
            visibility = 1.0

            p_left = smooth_left[i]
            p_right = smooth_right[i]
            eye_dist = float(np.linalg.norm(p_right - p_left))
            normalized_face_size = eye_dist / img_w

            if occlusion_culling:
                if normalized_face_size < fade_threshold:
                    visibility = max(0.0, (normalized_face_size - (fade_threshold / 2)) / (fade_threshold / 2))

            if visibility <= 0.0:
                out_images.append(bg_np[i])
                out_filter_only.append(np.zeros((img_h, img_w, 4), dtype=bg_np[i].dtype))
                continue

            use_3d = (perspective == "3D") and valid_3d[i]
            if use_3d:
                dst_pts = self._project_3d(
                    smooth_3d_left[i], smooth_3d_right[i], smooth_3d_nose[i], raw_intrinsics[i],
                    offset_x, offset_y, offset_z, scale_x, scale_y, tilt, rot_offset_x, rot_offset_y,
                )
            else:
                # 2D billboard. In 3D mode the texture was not pre-rotated, so the
                # tilt is applied here as a 2D rotation of the destination quad.
                extra_tilt = 0.0 if perspective == "2D" else tilt
                dst_pts = self._project_2d(
                    p_left, p_right, offset_x, offset_y, scale_x, scale_y, extra_tilt,
                )

            if dst_pts is None:
                out_images.append(bg_np[i])
                out_filter_only.append(np.zeros((img_h, img_w, 4), dtype=bg_np[i].dtype))
                continue

            src_pts = np.array([[0, 0], [f_w, 0], [f_w, f_h], [0, f_h]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            fg_warped = cv2.warpPerspective(
                (filter_np * 255).astype(np.uint8), M, (img_w, img_h),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
            )
            alpha_mask = (fg_warped[:, :, 3:4] / 255.0) * visibility
            fg_rgb = fg_warped[:, :, 0:3]
            frame_out = (fg_rgb * alpha_mask + frame_bg * (1.0 - alpha_mask)).astype(np.uint8)
            out_images.append(frame_out.astype(np.float32) / 255.0)
            out_filter_only.append((fg_warped * visibility).astype(np.float32) / 255.0)

        return (torch.from_numpy(np.array(out_images)), torch.from_numpy(np.array(out_filter_only)))


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "ApplyComfychatFilterMHR": ApplyComfychatFilterMHR,
    "ApplyComfychatFilterMediaPipe": ApplyComfychatFilterMediaPipe,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyComfychatFilterMHR": "Apply Comfychat Filter (MHR)",
    "ApplyComfychatFilterMediaPipe": "Apply Comfychat Filter (MediaPipe)",
}
