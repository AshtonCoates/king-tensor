
def main():
    blue = "\033[94m"
    red = "\033[91m"
    green = "\033[92m"
    reset = "\033[0m"

    print(
        f"{blue}kt run{reset}: "
        f"{green}launch the main King Tensor workflow (same as running kt with no args).{reset}"
    )
    print(
        f"{blue}kt boxer{reset} {red}<img_path>{reset}: "
        f"{green}interactively draw a box on img and get the resulting x, y, w, h.{reset}"
    )
    print(
        f"{blue}kt calibrate{reset} {red}[-f] [-t seconds] [-o output.yaml]{reset}: "
        f"{green}capture a screenshot and calibrate the screen (use -f for every UI element, -t to delay).{reset}"
    )
    print(
        f"{blue}kt dataclean clean-names{reset} {red}<png_dir>{reset}: "
        f"{green}rename non-numbered PNGs to 0001.png, 0002.png...{reset}"
    )
    print(
        f"{blue}kt dataclean sort{reset} {red}<csv_path> <image_dir> <output_dir>{reset}: "
        f"{green}read image/label columns, find images by filename (or suffix after last '-'), "
        f"and copy into label folders.{reset}"
    )
    print(
        f"{blue}kt help{reset}: "
        f"{green}show this overview of bundled utilities.{reset}"
    )
