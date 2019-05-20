FROM jupyter/scipy-notebook:ae5f7e104dd5

# First install some missing dependencies (namely, brian2)
RUN conda install brian2
