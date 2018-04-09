# absa
Aspect Based Sentiment Analysis Platform

python ../absa/manage.py shell_plus --notebook


# Celery
# run worker
$ celery -A absa worker --loglevel=warning -E

# run flower
$ celery flower -A absa --port=5555