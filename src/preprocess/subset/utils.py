# -*- coding: utf-8 -*-      
#--------------------------------------------
# Author:chenhao
# Date:2020-12-17 16:47
# Description:  
#--------------------------------------------
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234, help="random seed for initialization.")
    parser.add_argument('--input_path', default=None, type=str, required=True, help="The train input data dir.")
    parser.add_argument('--output_path', default=None, type=str, required=True, help="The output of checkpoint dir.")

    args = parser.parse_args()

    return args