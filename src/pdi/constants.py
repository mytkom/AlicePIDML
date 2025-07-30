"""This module contains project constants used in the experiments.
"""

TARGET_CODE_TO_PART_NAME = {
    11: "electron",
    13: "muon",
    211: "pion",
    321: "kaon",
    2212: "proton",
    -11: "antielectron",
    -13: "antimuon",
    -211: "antipion",
    -321: "antikaon",
    -2212: "antiproton",
}

# Reverse dict
PART_NAME_TO_TARGET_CODE = { v: k for k,v in TARGET_CODE_TO_PART_NAME.items() }

# Only target codes (particle species' PDG codes) of our interes
TARGET_CODES = [211, 2212, 321, -211, -2212, -321]
