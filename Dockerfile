FROM python:3

ADD ./requirements.txt /usr/src/gitlab-ci-series/requirements.txt
ADD ./compile_to_discord.sh /usr/src/gitlab-ci-series/compile_to_discord.sh

RUN chmod +x compile_to_discord.sh
RUN pip install -r /usr/src/gitlab-ci-series/requirements.txt

