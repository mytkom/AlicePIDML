"""This module contains project constants used in the experiments.
"""

PARTICLES_DICT = {
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

# Only target codes (particle species' PDG codes) of our interes
TARGET_CODES = [211, 2212, 321, -211, -2212, -321]
