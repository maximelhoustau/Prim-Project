#!/bin/sh

#Building Openpose

cd

wget -q https://cmake.org/files/v3.13/cmake-3.13.0-Linux-x86_64.tar.gz

sudo tar xfz cmake-3.13.0-Linux-x86_64.tar.gz --strip-components=1 -C /usr/local

git clone -q --depth 1 https://github.com/CMU-Perceptual-Computing-Lab/openpose.git

sed -i 's/execute_process(COMMAND git checkout master WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}\/3rdparty\/caffe)/execute_process(COMMAND git checkout f019d0dfe86f49d1140961f8c7dec22130c83154 WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}\/3rdparty\/caffe)/g' openpose/CMakeLists.txt

sudo apt-get -qq install -y libatlas-base-dev libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler libgflags-dev libgoogle-glog-dev liblmdb-dev opencl-headers ocl-icd-opencl-dev libviennacl-dev

sudo apt-get install libopencv-dev

sudo apt-get install cmake libblkid-dev e2fslibs-dev libboost-all-dev libaudit-dev

sudo sed -i '74d' /usr/local/cuda/include/crt/common_functions.h

cd openpose && rm -rf build || true && mkdir build && cd build && cmake .. && make -j`nproc`

#Get the data
cd

mkdir Videos
mkdir output


gsutil cp gs://projet-telecom-2020/*.mp4 ./Videos/
