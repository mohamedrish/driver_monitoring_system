import time
import winsound

class AlertManager:
    def __init__(self, min_interval_sec=1.0):
        self.min_interval_sec = min_interval_sec
        self.last_alert_time = 0.0

    def _beep(self, freq=880, duration=180):
        # duration is milliseconds
        winsound.Beep(int(freq), int(duration))

    def alert(self, alert_type: str):
        now = time.time()
        if now - self.last_alert_time < self.min_interval_sec:
            return
        self.last_alert_time = now

        if alert_type == "DROWSINESS":
            self._beep(freq=650, duration=220)
            time.sleep(0.05)
            self._beep(freq=650, duration=220)
        elif alert_type == "YAWN":
            self._beep(freq=520, duration=260)
        elif alert_type == "DISTRACTION":
            self._beep(freq=900, duration=120)
            time.sleep(0.03)
            self._beep(freq=900, duration=120)
            time.sleep(0.03)
            self._beep(freq=900, duration=120)
        elif alert_type == "PHONE":
            self._beep(freq=780, duration=180)
        else:
            self._beep()