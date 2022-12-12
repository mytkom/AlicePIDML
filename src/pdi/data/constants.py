COLUMNS_TO_SCALE = [
    "fTPCSignal",
    "fTRDSignal",
    "fTOFSignal",
    "fBeta",
    "fX",
    "fY",
    "fZ",
    "fAlpha",
    "fTPCNClsShared",
    "fDcaXY",
    "fDcaZ",
]
CSV_DELIMITER = ","
DROP_COLUMNS = ["fPdgCode", "fIsPhysicalPrimary", "P"]
INPUT_PATH = "data/raw/train_246_mc_multiple_detectors.csv"
MISSING_VALUES = {
    "fBeta": -999,
    "fTRDPattern": 0,
}
N_COLUMNS = 19
PROCESSED_DIR = "data/processed"
SEED = 42
TARGET_COLUMN = "fPdgCode"
TEST_SIZE = 0.3
TRAIN_SIZE = 0.55
GROUP_ID_KEY = "group_id"
