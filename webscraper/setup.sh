#!/bin/bash

pip install git+https://github.com/stanfordio/truthbrush.git
export TRUTHSOCIAL_USERNAME=aditya_murthy
export TRUTHSOCIAL_PASSWORD=Truthsocialus3r

verify_installation=$(which truthbrush 2>&1)

if [ $? -eq 0 ]; then
    echo "successfully installed"
else
    echo "$verify_installation"
fi

verify_working=$(truthbrush tags 2>&1)

if [ $? -eq 0 ]; then
    echo "successfully installed"
else
    echo "$verify_working"
fi