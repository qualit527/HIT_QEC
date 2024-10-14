.PHONY: build clean

default: build

build:
	python setup.py bdist_wheel

install: build
	python setup.py develop

clean:
	-pip uninstall bpdecoupling
	-rmdir /S /Q dist build
	-rmdir /S /Q "src\bpdecoupling.egg-info"
	-del "src\*.pyd"