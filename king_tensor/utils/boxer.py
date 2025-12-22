import sys
import cv2


def main(argv: list[str] | None = None):
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("Usage: kt boxer path/to/image")
        sys.exit(1)

    image_path = args[0]
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: could not load image '{image_path}'")
        sys.exit(1)

    # Window name
    win_name = "Select ROI - press ENTER/SPACE to confirm, C to cancel"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # Use built-in selection tool (drag with mouse)
    # returns (x, y, w, h)
    roi = cv2.selectROI(win_name, img, showCrosshair=True, fromCenter=False)

    cv2.destroyAllWindows()

    x, y, w, h = roi
    if w == 0 or h == 0:
        # User cancelled (or selected nothing)
        print("No ROI selected.")
        sys.exit(0)

    # Print as plain text so you can pipe/parse easily
    print(f"{x} {y} {w} {h}")

if __name__ == "__main__":
    main()
