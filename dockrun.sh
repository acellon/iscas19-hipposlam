#!/bin/bash
# Shell script to run Brian Docker file

_cwd="$PWD"
_container="${1:-${_cwd##*/}}"

docker run -it -p 8888:8888 -v $_cwd/src:/home/jovyan/work:rw $_container start-notebook.sh --NotebookApp.contents_manager_class="jupytext.TextFileContentsManager" --ContentsManager.default_jupytext_formats="ipynb,py:percent"
