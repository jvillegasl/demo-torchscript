#!/bin/bash

source ~/.bashrc

SCRIPTPATH=$PWD

export-model() {
	python "${SCRIPTPATH}/model/main.py" --export
}

latest-model() {
	local model=`ls ${SCRIPTPATH}/model/build/*.pt | sort -r | head -n 1`
	echo $model
}

clear-cpp() {
	rm -rf ${SCRIPTPATH}/cpp/build
}

build-cpp() {
	local initial_path=$PWD
	
	cd ${SCRIPTPATH}/cpp
	mkdir -p build
	cd build

	cmake ..
	cmake --build . --config Release

	cd $initial_path
}

pred() {
	local model_path

	if test -n "$1"; then
		model_path=$1
	elif test ! -t 0; then
		read line
		model_path=$line
	fi

	${SCRIPTPATH}/cpp/build/Release/simple-torchscript.exe $model_path
}
