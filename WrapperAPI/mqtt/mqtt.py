import paho.mqtt.client as mqtt
import time


class Mqtt:
    __MQTT_HOST = "localhost"
    __MQTT_USER = "tgr_user"
    __MQTT_PASS = "tgr_pass"
    __SUB_TOPIC = "led/status"
    __PUB_TOPIC = "led/control"

    def __init__(self):
        self.client = mqtt.Client(client_id="RASP-PI")
        self.client.username_pw_set(username="tgr_user", password="tgr_pass")
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(host=self.__MQTT_HOST)
        self.client.loop_start()

    def on_connect(self, client, userdata, flags, rc):
        print("--> Mqtt has been connected")
        self.client.subscribe(self.__SUB_TOPIC)

    def on_message(self, client, userdata, msg):
        #print("--> Mqtt on message")
        return

    def loop_stop(self):
        self.client.loop_stop()

    def toggle_led(self):
        self.client.publish(self.__PUB_TOPIC, "toggle1")
        print("Sent toggle LED1 successed")