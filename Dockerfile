FROM python:3

ADD ./requirements.txt /usr/src/gitlab-ci-series/requirements.txt

RUN pip install -r /usr/src/gitlab-ci-series/requirements.txt
