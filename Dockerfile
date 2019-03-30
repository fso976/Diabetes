FROM ubuntu:16.04
MAINTAINER Frank So <https://github.com/fso976/final_project>

RUN apt-get update
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install numpy pandas matplotlib sklearn seaborn plotly

ENTRYPOINT ["python3","diabetesfinal.py"]

