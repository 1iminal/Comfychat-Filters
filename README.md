# Comfychat-Filters

Custom ComfyUI nodes for overlaying images on top of faces, Snapchat-style.

| 2D (MediaPipe) | 3D (SAM3D Body) |
|----------------|-----------------|
|                |                 |

WAN identity/expression drift negation for 2D characters (Proposal)

| Default behavior (identity/expression drift) | Faceless + Comfychat-Filters |
|----------------------------------------------|------------------------------|
|                                              |                              |

## Installation

### With ComfyUI-Manager

1. Install [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager).
2. Look up **Comfychat-Filters** in ComfyUI-Manager and install it.
3. Restart ComfyUI.

### Manually

1. Clone this repository into `ComfyUI\custom_nodes`:

   ```bat
   git clone https://github.com/1iminal/Comfychat-Filters.git ComfyUI\custom_nodes\Comfychat-Filters
   ```

2. Install the dependencies:

   ```bat
   pip install -r requirements.txt
   ```

   For ComfyUI portable:

   ```bat
   python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\Comfychat-Filters\requirements.txt
   ```

3. Restart ComfyUI.

## Nodes

### Apply Comfychat Filter (MHR)

Adds a Snapchat filter-like image overlay to input images containing human faces, driven by SAM3D Body predicted pose keypoints.
Using SAM3D Body is a relatively heavy and slow approach, but it has the benefit of working realiably with non-realistic (anime, cartoon, 3D) style input.

#### Inputs

| Parameter          | Type          | Description                                                      |
|--------------------|---------------|------------------------------------------------------------------|
| `mhr_pose_data`    | MHR_POSE_DATA | SAM3D Body prediction result                                     |
| `background_image` | IMAGE         | Image to overlay the filter on                                   |
| `filter_image`     | IMAGE         | Filter image with alpha. 1:1 aspect ratio recommended. Image must be joined with its alpha (mask) using the built-in "Join Image With Alpha" node |
| `offset_x`         | FLOAT         | Horizontal filter offset                                         |
| `offset_y`         | FLOAT         | Vertical filter offset                                           |
| `offset_z`         | FLOAT         | Filter depth (3D perspective only)                               |
| `scale_x`          | FLOAT         | Filter scale along X axis                                        |
| `scale_y`          | FLOAT         | Filter scale along Y axis                                        |
| `tilt`             | FLOAT         | Filter tilt along its center (-180 to 180)                       |
| `rot_offset_x`     | FLOAT         | (3D perspective only) Filter rotation along X axis (-180 to 180); Rotations are applied in X-Y-Z(tilt) order.        |
| `rot_offset_y`     | FLOAT         | (3D perspective only) Filter rotation along Y axis (-180 to 180); Rotations are applied in X-Y-Z(tilt) order.        |
| `smoothing_window` | INT           | Number of sampled adjacent frames for motion smoothing (jitter reduction); 1 - no smoothing, 3-7 - optimal for video |
| `perspective`      | select        | 2D - filter always faces the camera, 3D - makes filter follow head rotation                                          |
| `occlusion_culling`| BOOLEAN       | Hide filter when the face is turned away or occluded                                                                 |
| `fade_threshold`   | FLOAT         | Distance between eyes at which filter begins to fade out (relative to the total width of the image)                  |

#### Outputs

| Output        | Type  | Description                                                   |
|---------------|-------|---------------------------------------------------------------|
| `IMAGE`       | IMAGE | The processed output image(s)                                 |
| `FILTER_ONLY` | IMAGE | Only the resulting filter image with a transparent background |

### Apply Comfychat Filter (MediaPipe)

Adds a Snapchat filter-like image overlay to input images containing human faces, driven by MediaPipe face landmarks.
MediaPipe is lightweight and fast, but it works exclusively with realistic input.

#### Inputs

| Parameter          | Type            | Description                                                      |
|--------------------|-----------------|------------------------------------------------------------------|
| `face_landmarks`   | FACE_LANDMARKS  | Face landmarks from Detect Face Landmarks (MediaPipe)            |
| `background_image` | IMAGE           | Image to overlay the filter on                                   |
| `filter_image`     | IMAGE           | Filter image with alpha. 1:1 aspect ratio recommended. Image must be joined with its alpha (mask) using the built-in "Join Image With Alpha" node |
| `face_index`       | INT             | Which detected face to use (0 = first)                           |
| `offset_x`         | FLOAT           | Horizontal filter offset                                         |
| `offset_y`         | FLOAT           | Vertical filter offset                                           |
| `offset_z`         | FLOAT           | Filter depth (3D perspective only)                               |
| `scale_x`          | FLOAT           | Filter scale along X axis                                        |
| `scale_y`          | FLOAT           | Filter scale along Y axis                                        |
| `tilt`             | FLOAT           | Filter tilt along its center (-180 to 180)                       |
| `rot_offset_x`     | FLOAT           | (3D perspective only) Filter rotation along X axis (-180 to 180); Rotations are applied in X-Y-Z(tilt) order.        |
| `rot_offset_y`     | FLOAT           | (3D perspective only) Filter rotation along Y axis (-180 to 180); Rotations are applied in X-Y-Z(tilt) order.        |
| `smoothing_window` | INT             | Number of sampled adjacent frames for motion smoothing (jitter reduction); 1 - no smoothing, 3-7 - optimal for video |
| `perspective`      | select          | 2D - filter always faces the camera, 3D - makes filter follow head rotation                                          |
| `occlusion_culling`| BOOLEAN         | Hide filter when the face is turned away or occluded                                                                 |
| `fade_threshold`   | FLOAT           | Distance between eyes at which filter begins to fade out (relative to the total width of the image)                  |

#### Outputs

| Output        | Type  | Description                                                   |
|---------------|-------|---------------------------------------------------------------|
| `IMAGE`       | IMAGE | The processed output image(s)                                 |
| `FILTER_ONLY` | IMAGE | Only the resulting filter image with a transparent background |

## AI Notice

This custom node pack has been created with the substantial help from Google's AI (Gemini) and Qwen3.8-27B. It has been reviewed and tested by the repository owner.
