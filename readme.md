
Docker Container for Emerging Models of Computation (Brian2)
============================================================

This folder should allow you to run a docker container that runs Python 3 and
Brian2 in a Jupyter Notebook environment. To get it up and running, you'll first
need to have Docker installed on your computer. Download a stable release for
your operating system here: <https://docs.docker.com/install/>. After going
through the installation procedure there, do the following:

1.  Download this folder to your computer (clone via git, download as a zip, etc.)
2.  Navigate to the folder.
3.  Build the Docker container (note the final dot): `docker build -t emc-container .`
4.  Run the container, making sure to open the correct port and mounting our
    local source folder as a volume:
    `docker run -it -p 8888:8888 -v /PATH/TO/DIR/emc-brian-docker/src:/home/jovyan/work:rw emc-container`
    -   _NB:_ replace `/PATH/TO/DIR/` with the absolute path to the folder on your
        machine, but don't change `/home/jovyan/work` (this is the path within
        the container)
5.  Open an internet browser and navigate to the url from the terminal - you
    should find the Jupyter notebook and everything should work!

If you have any issues with the above, you may have to run the code as root
(i.e. append `sudo` to the front of your commands). After the first time you use
build the container, you should only have to do steps 4 and 5 above to get back
to where you left off!
