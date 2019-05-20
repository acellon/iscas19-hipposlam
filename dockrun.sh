#!/bin/bash
# Shell script to run Brian Docker file

_cwd="$PWD"
_container="${1:-${_cwd##*/}}"

sudo docker run -it -p 8888:8888 -v /home/adamc/Documents/research/iscas19-hipposlam/src:/home/jovyan/work:rw iscas19-hipposlam start-notebook.sh --NotebookApp.contents_manager_class="jupytext.TextFileContentsManager" --ContentsManager.default_jupytext_formats="ipynb,py:percent"
