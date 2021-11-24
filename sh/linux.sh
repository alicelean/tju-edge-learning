#!/bin/bash

#echo "开始测试......"
n_nodes=5
int=1
while(( $int<=$n_nodes ))
do
    /Users/alice/opt/anaconda3/envs/tensorflow/bin/python  /Users/alice/tju.com/python/tju-edge-learning/client_cnn.py&
     let "int++"
done