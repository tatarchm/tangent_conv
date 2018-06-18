import argparse
parser = argparse.ArgumentParser(description='Tangent convolutions')
parser.add_argument('input_folder', type=str, metavar='input_folder', help='dataset folder')
parser.add_argument('output_folder', type=str, metavar='output_folder', help='output folder')
parser.add_argument('dataset', type=str, metavar='dataset', help='dataset type')
args = parser.parse_args()

from util.path_config import *
from util.dataset_params import *
from util.cloud import *
from util.common import *

import os
import csv
import json
import wget

def get_stanford():
	class_dict = {"ceiling" : 1, "floor" : 2, "wall" : 3, "beam" : 4, "column" : 5,
				  "window" : 6, "door" : 7, "table" : 8, "chair" : 9, "sofa" : 10,
				  "bookcase" : 11, "board" : 12, "clutter" : 13, "stairs" : 14}

	zip_file_name = os.path.join(args.input_folder, "Stanford3dDataset_v1.2_Aligned_Version.zip")
	cmd = "unzip " + zip_file_name + " -d " + args.input_folder
	os.system(cmd)

	print("Converting to PCD...")
	areas = ["Area_1", "Area_2", "Area_3", "Area_4", "Area_5", "Area_6"]
	for area in areas:
		print(":: " + area)
		scans = list_dir_single(os.path.join(os.path.splitext(zip_file_name)[0], area))
		for scan in scans:
			print(scan)
			points = []
			colors = []
			labels = []

			out_dir_name = os.path.join(args.output_folder, area + "_" + scan)
			make_dir(out_dir_name)

			scan_path = os.path.join(os.path.splitext(zip_file_name)[0], area, scan, "Annotations")
			objects = list_files_single(scan_path)
			for obj in objects:
				if os.path.splitext(obj)[1] != ".txt":
					continue
				with open(os.path.join(scan_path, obj)) as f:
					entries = f.readlines()
					for entry in entries:
						if len(entry) < 2:
							continue
						
						spl = entry.split()
						
						if len(spl) != 6:
							continue

						res = [float(c) for c in spl]
						points.append(np.asarray(res[0:3]))
						colors.append(np.asarray(res[5:6] + res[4:5] + res[3:4]) / 255.0)

					class_name = obj.split("_")[0]
					labels += [class_dict[class_name]] * len(entries)

			pcd = PointCloud()
			pcd.points = Vector3dVector(points)
			pcd.colors = Vector3dVector(colors)
			write_point_cloud(os.path.join(out_dir_name, "scan.pcd"), pcd)
			with open(os.path.join(out_dir_name, "scan.labels"), "w") as f:
				f.writelines(["%s\n" % item for item in labels])
	print("Done.")

def get_scannet():
	class_dict = {"1" : 1, "2" : 2, "3" : 3, "4" : 4, "5" : 5,
		 		  "6" : 6, "7" : 7, "8" : 8, "9" : 9, "10" : 10,
		  		  "11" : 11, "12" : 12, "14" : 13, "16" : 14,
		  		  "24" : 15, "28" : 16, "33" : 17, "34" : 18,
		  	   	  "36" : 19, "39" : 20}

	label_map = {}
	label_map_file = os.path.join(args.input_folder, "scannet-labels.combined.tsv")
	with open(label_map_file, 'r') as f:
		lines = csv.reader(f, delimiter='\t')
		cnt = 0
		for line in lines:
			if cnt == 0:
				print(line)
			else:
				if len(line[4]) > 0:
					label_map[line[1]] = line[4]
				else:
					label_map[line[1]] = '0'
			cnt += 1

	print("Converting to PCD...")
	for room_name in list_dir_single(args.input_folder):
		print(room_name)
		aggregation_file = os.path.join(args.input_folder, room_name, room_name + ".aggregation.json")
		seg_file = os.path.join(args.input_folder, room_name, room_name + "_vh_clean_2.0.010000.segs.json")
		ply_file = os.path.join(args.input_folder, room_name, room_name + "_vh_clean_2.ply")
		pcd = read_point_cloud(ply_file)

		ca = np.asarray(pcd.colors)
		ca = np.concatenate((ca[:,2:3], ca[:,1:2], ca[:,0:1]), axis=1)
		pcd.colors = Vector3dVector(ca)

		with open(aggregation_file) as f:
			aggregation_data = json.load(f)

		with open(seg_file) as f:
			seg_data = json.load(f)

		str_segments = seg_data["segIndices"]
		int_segments = np.asarray(str_segments, dtype='int32')
		out_labels = np.zeros((len(int_segments)), dtype='int32')

		num_objects = len(aggregation_data["segGroups"])
		for obj in aggregation_data["segGroups"]:
			str_lbl = obj["label"]
			for seg in obj["segments"]:
				int_seg = int(seg)
				ind = int_segments == int_seg
				if str_lbl in label_map:
					lb = label_map[str_lbl]
				else:
					lb = '-'
				if lb in class_dict.keys():
					out_labels[ind] = class_dict[lb]
				else:
					out_labels[ind] = 0

		out_dir_name = os.path.join(args.output_folder, room_name)
		make_dir(out_dir_name)
		with open(os.path.join(out_dir_name, "scan.labels"), "w") as f:
			f.writelines(["%s\n" % item  for item in out_labels])
		write_point_cloud(os.path.join(out_dir_name, "scan.pcd"), pcd)
	print("Done.")

def get_semantic3d():
	base_url = "http://www.semantic3d.net/data/point-clouds/training1/"
	dl_files = {"bildstein_station1" : "bildstein_station1_xyz_intensity_rgb",
				"bildstein_station3": "bildstein_station3_xyz_intensity_rgb",
				"bildstein_station5" : "bildstein_station5_xyz_intensity_rgb",
				"domfountain_station1" : "domfountain_station1_xyz_intensity_rgb",
				"domfountain_station2" : "domfountain_station2_xyz_intensity_rgb",
				"domfountain_station3" : "domfountain_station3_xyz_intensity_rgb",
				"neugasse_station1" : "neugasse_station1_xyz_intensity_rgb",
				"sg27_station1" : "sg27_station1_intensity_rgb",
				"sg27_station2" : "sg27_station2_intensity_rgb",
				"sg27_station4" : "sg27_station4_intensity_rgb",
				"sg27_station5" : "sg27_station5_intensity_rgb",
				"sg27_station9" : "sg27_station9_intensity_rgb",
				"sg28_station4" : "sg28_station4_intensity_rgb",
				"untermaederbrunnen_station1" : "untermaederbrunnen_station1_xyz_intensity_rgb",
				"untermaederbrunnen_station3" : "untermaederbrunnen_station3_xyz_intensity_rgb"}

	labels_url = "http://www.semantic3d.net/data/sem8_labels_training.7z"
	print("Downloading...")
	for dl_file in dl_files:
		print("")
		dl_file_path = os.path.join(args.input_folder, dl_files[dl_file] + ".7z")
		print(dl_file)
		if not os.path.exists(dl_file_path):
			wget.download(base_url + dl_files[dl_file] + ".7z",
						  out=dl_file_path)

	labels_file = os.path.join(args.input_folder, "labels.7z")
	if not os.path.exists(labels_file):
		wget.download(labels_url, out=labels_file)
	print("done.")

	print("Extracting...")
	for key in dl_files:
		dl_file_path = os.path.join(args.input_folder, key)
		if not os.path.exists(dl_file_path):
			inner_path = dl_files[key]
			if key == "neugasse_station1":
				inner_path = "station1_xyz_intensity_rgb"
			cmd = "7z x " + os.path.join(args.input_folder, dl_files[key] + ".7z") + " -o" + dl_file_path
			os.system(cmd)
			cmd = "mv " + os.path.join(dl_file_path, inner_path + ".txt") + " " + os.path.join(dl_file_path, "scan.txt")
			print(cmd)
			os.system(cmd)

	cmd = "7z x " + os.path.join(args.input_folder, "labels.7z") + " -o" + os.path.join(args.input_folder, "labels")
	os.system(cmd)

	for key in dl_files:
		cmd = "mv " + os.path.join(args.input_folder, "labels", dl_files[key] + ".labels") + " " + os.path.join(args.input_folder, key, "scan.labels")
		os.system(cmd)
	print("done.")

	for key in dl_files:
		
		points = []
		colors = []
		labels = []

		with open(os.path.join(args.input_folder, key, "scan.txt")) as f:
			cnt = 0
			entries = f.readlines()
			for entry in entries:
				res = [float(c) for c in entry.split()]
				points.append(np.asarray(res[0:3]))
				colors.append(np.asarray(res[6:7] + res[5:6] + res[4:5]) / 255.0)
				if cnt % 100000 == 0:
					print(cnt)
				cnt += 1
		
		os.mkdir(os.path.join(args.output_folder, key))
		pcd = PointCloud()
		pcd.points = Vector3dVector(points)
		pcd.colors = Vector3dVector(colors)
		write_point_cloud(os.path.join(args.output_folder, key, "scan.pcd"), pcd)

		cmd = "cp " + os.path.join(args.input_folder, key, "scan.labels") + " " + os.path.join(args.output_folder, key, "scan.labels")
		os.system(cmd)



if args.dataset == "stanford":
	get_stanford()
elif args.dataset == "scannet":
	get_scannet()
elif args.dataset == "semantic3d":
	get_semantic3d()
else:
	print("Wrong dataset type")