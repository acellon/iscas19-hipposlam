FROM jupyter/scipy-notebook:ae5f7e104dd5

# First install some necessary dependencies (already using conda-forge channel)
RUN conda install brian2
RUN conda install jupytext

# RUN jupyter notebook --generate-config

# RUN jupyter notebook --generate-config #&& \
    # echo 'c.NotebookApp.token = u""' >> ~/.jupyter/jupyter_notebook_config.py && \
    #echo 'c.NotebookApp.contents_manager_class = "jupytext.TextFileContentsManager"' >> ~/.jupyter/jupyter_notebook_config.py && \
    #echo 'c.ContentsManager.default_jupytext_formats = "ipynb,py"' >> ~/.jupyter/jupyter_notebook_config.py
