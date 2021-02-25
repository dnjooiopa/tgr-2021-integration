from apscheduler.schedulers.background import BackgroundScheduler
import time
import requests
from datetime import datetime

url = '<SERVER_IP>'

def heartbeat():
    r = requests.get(url)
    if int(r.status_code) == 200:
        print('❤❤❤❤❤', end=" ")
        print('<', datetime.fromtimestamp(int(r.text)), '>', end=" ")
        print('❤❤❤❤❤')
    else:
        print('😢😢😢😢😢', end=" ")
        print('<', datetime.fromtimestamp(int(r.text)), '>', end=" ")
        print('😢😢😢😢😢')

scheduler = BackgroundScheduler()
scheduler.add_job(heartbeat, 'interval', minutes=1)
scheduler.start()
print("<---start checking heartbeat--->")
heartbeat()
try:
    # This is here to simulate application activity (which keeps the main thread alive).
    while True:
        time.sleep(2)
except (KeyboardInterrupt, SystemExit):
    # Not strictly necessary if daemonic mode is enabled but should be done if possible
    scheduler.shutdown()