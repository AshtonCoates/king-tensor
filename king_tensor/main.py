import mss
import yaml
from .screen import Screen
from .gamestate import Gamestate


def main():
    # load configs
    print("okay boomer")

    with mss.mss() as sct:
        # Grab the first monitor (index 0 is usually all monitors combined)
        monitor_1 = sct.monitors[0]
        sct.grab(monitor_1)
        # The image data is accessible via the sct.screenshot object or can be saved directly
        sct.save(f"screenshot.png")
