#!/bin/bash

source ~/.bashrc

export-model() {
	python model/main.py --export
}

latest-model() {
	local model=`ls model/build/*.pt | sort -r | head -n 1`
	echo $PWD/$model
}

build-cpp() {
	cd cpp
	mkdir -p build
	cd build

	cmake ..
	cmake --build . --config Release

	cd ../..
}

pred() {
	local model_path

	if test -n "$1"; then
		model_path=$1
	elif test ! -t 0; then
		read line
		model_path=$line
	fi

	./cpp/build/Release/simple-torchscript.exe $model_path
}
