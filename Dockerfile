FROM python:3.7.6-slim

ADD ./requirements.txt 

RUN pip install -r /usr/src/gitlab-ci-series/requirements.txt
