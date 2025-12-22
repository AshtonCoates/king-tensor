import mss
import numpy as np
import yaml
from .screen import Screen
from .gamestate import Gamestate
from PIL import Image
from pathlib import Path
from king_tensor.globals import CONFIG_DIR

def parse_config(path: Path):
    with open(path, 'r') as config_file:
        config_dict = yaml.safe_load(config_file)
        return config_dict

def main():
    # load configs
    screen_config_path = CONFIG_DIR / 'alt.yaml'
    gamestate_config_path = CONFIG_DIR / 'gamestate.yaml'

    screen_config = parse_config(screen_config_path)
    gamestate_config = parse_config(gamestate_config_path)

    with mss.mss() as sct:
        monitor = sct.monitors[1]
        img = sct.grab(monitor)
        #mss.tools.to_png(img.rgb, img.size, output='screenshot.png')
        pil_img = Image.frombytes(
            "RGB",
            img.size,
            img.rgb
        )
        img = np.array(pil_img)
    
    screen = Screen(img, screen_config)
    screen.plot_images()
