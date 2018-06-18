
# JSON configuration parameters

## Precomputation

- ``pre_min_cube_size``: Voxel size at the lowest resultion. It is doubled at each downsampling stage.
- ``pre_filter_size``: Number of elements in each dimension of the convolutional kernel.
- ``pre_num_scales``: The number of resolutions to use.
- ``pre_num_rotations``: Number of rotations used for rotational augmentation.
- ``pre_num_neighbors``: Number of neighboring points used to precompute the structure of conv kernel.
- ``pre_noise_level``: Standard deviation of the additive Gaussian noise on point locations.
- ``pre_output_dir``: Directory for storing data after pre-computation.

- ``pre_interp_method``: Method for interpolating the signal in tangent images ('depth_densify_nearest_neighbor', 'depth_densify_gaussian_kernel').
- ``pre_dataset_param``: Dataset type ('stanford', 'scannet' or 'semantic3d').

## Common

- ``co_train_file``: List of training scans.
- ``co_test_file``: List of test scans.
- ``co_experiment_dir``: Path to current experiment directory.
- ``co_output_dir``: Relative path to store network outputs.

## Training and testing

- ``tt_log_dir``: Directory where to output logs.
- ``tt_snapshot_dir``: Directory for saving network snapshots.
- ``tt_input_type``: Which input features to use for training. A string containing one or more of the following values: ``c`` (color), ``d`` (depth), ``n`` (normals), ``h`` (height).
- ``tt_max_snapshots``: Maximum number of snapshots to be saved.
- ``tt_test_iter``: Frequency of running the validation during training (in iterations).
- ``tt_reload_iter``: Frequrency of updating the training set (in iterations). Used because of the rotational augmentation, to not store all scans in memory all the time.
- ``tt_max_iter_count``: Maximum number of training iterations.
- ``tt_batch_size``: Batch size.
- ``tt_valid_rad``: Radius of a sphere to be used for sampling when working with large scans.
- ``tt_filter_size``: Size of the convolutional filter along one dimension.
- ``tt_batch_array_size``: Number of batches to pre-load when sampling from large scans.

## Evaluation

- ``eval_scan_file``: Name pattern for the raw scan file.
- ``eval_label_file``: Name pattern for the raw label file.
- ``eval_output_file``: Name pattern for file with extrapolated labels.