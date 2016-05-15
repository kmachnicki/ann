#!/bin/bash

echo "***** Installing required Python packages *****"

pip3 install -r requirements.txt

echo "***** Downloading required Python-ELM module *****"

if ! [ -d "modules" ] ; then
	mkdir "modules"
fi

git clone https://github.com/kmachnicki/Python-ELM modules/Python-ELM

echo "***** All done, enjoy *****"
