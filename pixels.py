import time
import random
import requests
from ast import literal_eval
from PIL import Image
from typing import List, Optional, Tuple

RL_DATA = Tuple[str, str, str, str]
RGB = Tuple[int, ...]

API_TOKEN = (
    "token lol"
)
AUTH_HEADER = {"Authorization": f"Bearer {API_TOKEN}"}

size_request = requests.get("https://pixels.pythondiscord.com/get_size", headers=AUTH_HEADER)
size_dict = literal_eval(size_request.content.decode("utf-8"))
WIDTH, HEIGHT = size_dict["width"], size_dict["height"]


class Area:
    """Specification for rectangular areas of pixels on the canvas."""

    def __init__(self, x1, y1, x2, y2, img=None):
        """Initialize an Area instance."""
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        self.img = img


def rgb_to_hex(rgb_tuple: RGB) -> str:
    """Return a zero-padded hex value string given an RGB tuple."""
    return "".join(f"{rgb:>02x}" for rgb in rgb_tuple)


def close_rgb(rgb_tuple_one: RGB, rgb_tuple_two: RGB, margin: int) -> bool:
    """Return if two RGB tuples are close to each other given a margin."""
    return abs(sum(rgb_tuple_one) - sum(rgb_tuple_two)) < margin


def load_img_matrix(img: str) -> List[List[RGB]]:
    im = Image.open(img)

    all_pixels_rgb = [rgb[0:3] for rgb in list(im.getdata())]
    return pixels_xy(all_pixels_rgb, im.width, im.height)


def pixels_xy(all_pixels_rgb: List[RGB], img_width: int, img_height: int) -> List[List[RGB]]:
    """
    Return a matrix of RGB values.

    Return a matrix allowing support for easy access of x-y coordinates
    from an original image data list containing RGB values.
    """
    return [
        all_pixels_rgb[(i-1)*img_width:i*img_width]
        for i in range(1, img_height + 1)
    ]


def wait_for_rl(rld: RL_DATA, message: str) -> None:
    """Wait for ratelimit reset to end before continuing the program if no allowed requests remain."""
    if int(rld[0]) == 0:
        print(message)
        time.sleep(float(rld[2]))


def get_ratelimit_data(r: requests.models.Response) -> RL_DATA:
    """Fetch ratelimit data."""
    headers = r.headers

    cd = headers.get("cooldown-reset", None)
    return (
        headers["requests-remaining"],
        headers["requests-limit"],
        headers["requests-reset"],
        cd,
    )


def get_pixels() -> Tuple[List[RGB], List[List[RGB]]]:
    """
    Return all the pixels on screen, in a raw version and a matrix version.

    Also return some ratelimit data.
    """
    pixels_resp = requests.get("https://pixels.pythondiscord.com/get_pixels", headers=AUTH_HEADER)
    rld = get_ratelimit_data(pixels_resp)
    wait_for_rl(rld, "Sleeping for /get_pixels endpoint's ratelimit...")

    pixels_data = pixels_resp.content

    all_pixels_rgb = [
        tuple(
            pixels_data[(i-1)*3:i*3][j]
            for j in range(3)
        )
        for i in range(1, WIDTH * HEIGHT + 1)
    ]

    all_pixels_rgb_xy = pixels_xy(all_pixels_rgb, WIDTH, HEIGHT)

    return all_pixels_rgb, all_pixels_rgb_xy


def set_pixel(x: int, y: int, hexstr: str) -> None:
    """Set a pixel to a specified colour."""
    pixel_data = {
        "x": x,
        "y": y,
        "rgb": hexstr,
    }

    post_resp = requests.post(
        "https://pixels.pythondiscord.com/set_pixel",
        json=pixel_data,
        headers=AUTH_HEADER,
    )

    rld = get_ratelimit_data(post_resp)
    wait_for_rl(rld, "Sleeping for /set_pixel endpoint's ratelimit...")

    post_resp.raise_for_status()
    print(post_resp.json()["message"])


def display_canvas(scale: int) -> None:
    """Show the current canvas in the default image-displaying software."""
    im = Image.new("RGB", (WIDTH, HEIGHT))
    im.putdata(get_pixels()[0])
    im = im.resize((WIDTH * scale, HEIGHT * scale), Image.NEAREST)
    im.show()


def get_rgb(x: int, y: int) -> RGB:
    """Get the RGB value of a certain pixel."""
    return get_pixels()[1][y][x]


def line_invasion(x_start: int, y: int) -> None:
    """Invade one column of pixels with randomized colors."""
    for i in range(x_start, WIDTH):
        set_pixel(
            i,
            HEIGHT - y,
            f"{random.randint(0, 16777215):>06x}"
        )


def copy_image(img: str, x_topleft: int, y_topleft: int, ignore_color: Optional[RGB] = None) -> None:
    """
    Copy an image to the canvas pixel by pixel.

    Can also be used for protection if ran repeatedly, but this is not recommended.
    """
    all_pixels_rgb_xy = load_img_matrix(img)

    for y, column in enumerate(all_pixels_rgb_xy, start=y_topleft):
        for x, rgb in enumerate(column, start=x_topleft):

            # Check if the targeted pixel was not already written with the RGB, or if it should be ignored
            if not close_rgb(rgb, get_rgb(x, y), 12) and rgb != ignore_color:
                set_pixel(x, y, rgb_to_hex(rgb))


def protect_areas(areas: List[Area], image_mode: bool = True) -> None:
    original_pixels_matrices = [load_img_matrix(area.img) for area in areas]
    original_pixels_matrix = get_pixels()[1]

    while True:
        current_pixels_matrix = get_pixels()[1]

        for area_num, area in enumerate(areas):
            for y, column in enumerate(current_pixels_matrix[area.y1:area.y2 + 1], start=area.y1):
                for x, current_rgb in enumerate(column[area.x1:area.x2 + 1], start=area.x1):
                    if image_mode:
                        original_rgb = original_pixels_matrices[area_num][y][x]
                    else:
                        original_rgb = original_pixels_matrix[y][x]

                    print(x, y, current_rgb, original_rgb)
                    if not close_rgb(current_rgb, original_rgb, 12):
                        set_pixel(x, y, rgb_to_hex(original_rgb))
