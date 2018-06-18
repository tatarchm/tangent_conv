import numpy as np

class stanford_params:
	def __init__(self):
		self.class_freq = np.asarray([19.203, 16.566, 27.329,
						 			  2.428, 2.132, 2.123, 5.494, 3.25,
						 			  4.079, 0.488, 4.726, 1.264, 10.918, 100.0])
		self.class_weights = -np.log(self.class_freq / 100.0)
		self.num_classes = len(self.class_freq) + 1
		self.color_map = [[255, 255, 255], # unlabeled (white)
 						  [128, 128, 128], # ceiling (red)
 						  [124, 152, 0], # floor (green)
 						  [255, 225, 25], # walls (yellow)
 						  [0,   130, 200], # beam (blue)
 						  [245, 130,  48], # column (orange)
 						  [145,  30, 180], # window (purple)
 						  [0, 130, 200], # door (cyan)
 						  [0, 0, 128], # table (black)
 						  [128, 0, 0], # chair (maroon)
 						  [250, 190, 190], # sofa (pink)
 						  [170, 110, 40], # bookcase (teal)
 						  [0, 0, 0], # board (navy)
 						  [170, 110,  40], # clutter (brown)
 						  [128, 128, 128]] # stairs (grey)

class scannet_params:
	def __init__(self):
		self.class_freq = np.asarray([40.82, 27.79, 2.96, 2.81, 5.04, 2.94, 2.81, 2.51, 1.06, 2.25,
			  			 0.42, 0.73, 1.86, 1.43, 0.46, 0.20, 0.3, 0.38, 0.36, 2.84])
		self.class_weights = -np.log(self.class_freq / 100.0)
		self.num_classes = len(self.class_freq) + 1
		self.color_map = [[  0,   0,   0], # unlabeled (white)
					 	  [190, 153, 112], # wall
					 	  [189, 198, 255], # floor
					 	  [213, 255,   0], # cabinet
					 	  [158,   0, 142], # bed
					 	  [152, 255,  82], # chair
					 	  [119,  77,   0], # sofa
					 	  [122,  71, 130], # table
					 	  [  0, 174, 126], # door
					 	  [  0, 125, 181], # window
					 	  [  0, 143, 156], # bookshelf
					 	  [107, 104, 130], # picture
					 	  [255, 229,   2], # counter
					 	  [  1, 255, 254], # desk
					 	  [255, 166, 254], # curtain
					 	  [232,  94, 190], # refridgerator
					 	  [  0, 100,   1], # shower curtain
					 	  [133, 169,   0], # toilet
					 	  [149,   0,  58], # sink
					 	  [187, 136,   0], # bathtub
					 	  [  0,   0, 255]] # otherfurniture (blue)

class semantic3d_params:
	def __init__(self):
		self.class_freq = np.asarray([41.227, 24.391, 6.845, 5.153, 14.673, 4.23, 2.7, 0.782])
		self.class_weights = -np.log(self.class_freq / 100.0)
		self.num_classes = len(self.class_freq) + 1
		self.color_map = [[255, 255, 255], # unlabeled (white)
			 			  [128, 128, 128], # man made terrain (grey)
			 			  [255, 225, 25],   # natural terrain (yellow)
			 			  [124, 152, 0], 	  # high vegetation (dark green)
			 			  [170, 110, 40],   # low vegetation (light green)
			 			  [128, 0, 0], 	  # building (red)
			 			  [245, 130, 48],   # hardscape (purple)
			 			  [250, 190, 190], # scanning artifacts (light blue)
			 			  [0, 130, 200]]  # cars (pink)