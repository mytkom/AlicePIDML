"""This module contains project constants used in the experiments.
"""

PARTICLES_DICT = {
    11: "electron",
    13: "mion",
    211: "pion",
    321: "kaon",
    2212: "proton",
    -11: "antielectron",
    -13: "antimion",
    -211: "antipion",
    -321: "antikaon",
    -2212: "antiproton",
}

PART_DICT = {
    11: "El",
    13: "Mu",
    211: "Pi",
    321: "Ka",
    2212: "Pr",
}

TARGET_CODES = [211, 2212, 321, -211, -2212, -321]
P_RANGE = [0, 5]
P_RESOLUTION = 20
NUM_WORKERS = 3
