FROM python:3

ADD ./requirements.txt /usr/src/gitlab-ci-series/requirements.txt

RUN chmod +x compile_to_discord.sh
RUN pip install -r /usr/src/gitlab-ci-series/requirements.txt
