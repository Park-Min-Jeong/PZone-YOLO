import os
os.system("sed -i 's/OPENCV=0/OPENCV=1/' Makefile")
os.system("sed -i 's/GPU=0/GPU=1/' Makefile")
os.system("sed -i 's/CUDNN=0/CUDNN=1/' Makefile")
os.system("sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile")
os.system("sed -i 's/LIBSO=0/LIBSO=1' Makefile")
os.system("make")
os.system("chmod +x ./darknet")
