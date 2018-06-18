import argparse
parser = argparse.ArgumentParser(description='Tangent convolutions')
parser.add_argument('config', type=str, metavar='N', help='config json file')
parser.add_argument('--precompute', action="store_true", help='do precomputation')
parser.add_argument('--train', action="store_true", help='do training network')
parser.add_argument('--test', action="store_true", help='do testing network')
parser.add_argument('--extrapolate', action="store_true", help='do extrapolate for the evaluation')
parser.add_argument('--evaluate', action="store_true", help='do evaluation')
args = parser.parse_args()

# reading configuration files
import os, sys
sys.path.append('util')
from config_reader import *
config = config_reader(args.config)

# do actions
if args.precompute:
	from precompute import *
	print(":: precompute")
	run_precompute(config)

if args.train:
	from model import *
	print(":: training")
	run_net(config, "train")

if args.test:
	from model import *
	print(":: testing")
	run_net(config, "test")

if args.extrapolate:
	from extrapolate import *
	print(":: extrapolate")
	run_extrapolate_labels(config)

if args.evaluate:
	from eval import *
	print(":: evaluate")
	run_eval(config)
