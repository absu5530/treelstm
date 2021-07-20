FROM python:3.8-slim-buster

VOLUME /data
WORKDIR /data

RUN apt-get update && apt-get install -y python3-pip

RUN pip3 install pandas==1.2.4 spacy==3.0.6 scikit-learn==0.24.1 mxnet==1.8.0.post0 gluonnlp==0.10.0

RUN python3 -m spacy download en_core_web_trf

ENTRYPOINT ["/bin/bash"] 
