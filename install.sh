#!/bin/bash

echo "***** Installing HDF5 (used by ELM) *****"

pacman -S hdf5

echo "***** Installing required Python packages *****"

pip3 install -r requirements.txt

echo "***** All done, enjoy *****"
