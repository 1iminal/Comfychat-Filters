# Apply Comfychat Filter (MHR)

Adds a Snapchat filter-like image overlay to input images containing human faces, driven by SAM3D Body predicted pose keypoints.
Using SAM3D Body is a relatively heavy and slow approach, but it has the benefit of working realiably with non-realistic (anime, cartoon, 3D) style input.

## Inputs

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

## Outputs

| Output        | Type  | Description                                                   |
|---------------|-------|---------------------------------------------------------------|
| `IMAGE`       | IMAGE | The processed output image(s)                                 |
| `FILTER_ONLY` | IMAGE | Only the resulting filter image with a transparent background |
