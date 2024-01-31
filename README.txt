To run experiments, configuration function must be called. This function prints Top-1 accuracy of the particular experiment for all query sets. Thus, it computes accuracies of query sets altogether. 

Inputs for the configuration function are:
-histogram type: "3D" or "per_channel" 
-color space: "RGB" or "HSV"
-grid_size: 1 (for non spatial grid experiments 1x1), 2 (for 2x2 grids), 4 (for 4x4 grids), 6 (for 6x6 grids), 8 (for 8x8 grids)
-quantization interval: 8, 16, 32, 64, 128 

Example run for an experiment using 3D color histogram, RGB color space, non spatial grid and quantization interval of 16:
configuration("3D", "RGB", 1, 16)

All experiments that are used in the homework are commented in the code and ready to be used. 