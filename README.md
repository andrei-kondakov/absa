# absa
Aspect Based Sentiment Analysis Platform

python ../absa/manage.py shell_plus --notebook


# Celery
# run worker
$ celery -A absa worker --loglevel=warning -E --concurrency=1 -Q train
$ celery -A absa worker --loglevel=warning -E --concurrency=1 -Q evaluation

# run flower
$ celery flower -A absa --port=5555 --persistent


# Installation
# Install Anaconda
# https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-16-04

conda update -n base conda
conda create --name absa