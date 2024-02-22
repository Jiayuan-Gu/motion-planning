from typing import Sequence

from bs4 import BeautifulSoup


def dump_srdf(disable_collisions: Sequence[dict], srdf_path):
    srdf = BeautifulSoup("<robot></robot>", "xml")
    for disable_collision in disable_collisions:
        tag = srdf.new_tag("disable_collisions")
        tag["link1"] = disable_collision["link1"]
        tag["link2"] = disable_collision["link2"]
        tag["reason"] = disable_collision.get("reason", "Default")
        srdf.robot.append(tag)

    srdf_str = srdf.prettify()
    with open(srdf_path, "w") as f:
        f.write(srdf_str)
