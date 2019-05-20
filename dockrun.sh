#!/bin/bash
# Shell script to run Brian Docker file

_cwd="$PWD"
_container="${1:-${_cwd##*/}}"

sudo docker run -it -p 8888:8888 -v $_cwd/src:/home/jovyan/work:rw $_container
