# Apply Comfychat Filter (MediaPipe)

Adds a Snapchat filter-like image overlay to input images containing human faces, driven by MediaPipe face landmarks.
MediaPipe is lightweight and fast, but it works exclusively with realistic input.

## Inputs

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

## Outputs

| Output        | Type  | Description                                                   |
|---------------|-------|---------------------------------------------------------------|
| `IMAGE`       | IMAGE | The processed output image(s)                                 |
| `FILTER_ONLY` | IMAGE | Only the resulting filter image with a transparent background |
