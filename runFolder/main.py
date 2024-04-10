import yaml
import sys 
import os
sys.path.append(os.path.abspath("../src/"))
from deeponet_trunk import *
from deeponet_branch import *
from predict import *


if __name__ == '__main__':
	
	stream = open("input.yaml", 'r')
	dictionary = yaml.safe_load(stream)
	# print(dictionary)
    # for key, value in dictionary.items():
    #     print (key + " : " + str(value))

    # Train the trunk network
	print("Begin trunk net training ...")
	train_trunknet(dictionary)

    # Train the branch network
	print("Begin branch net training ...")
	train_branchnet(dictionary)

	# Predict the solutions
	infer_solution(dictionary)



