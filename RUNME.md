
Instructions on Docker:



First, install 'diabetesfinal.py' and 'Dockerfile' files on local drive

Build image:
docker build -t <image_name>

Run image:
docker run -ti -v ${PWD}:${PWD} -w ${PWD} <image_name>

All figures are saved in the ./Figures folder



