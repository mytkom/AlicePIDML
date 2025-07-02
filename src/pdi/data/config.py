# simulated data source
INPUT_PATH = "sim_csv_file_path"
# real data source for DANN
EXPERIMENTAL_INPUT_PATH = "exp_csv_file_path"
RUN = 3
UNDERSAMPLE = True # only for feature set preparation
MODEL_NAME = "model_name"
GET_NSIGMA = True
if RUN not in [2, 3]:
    raise ValueError("RUN must be 2 or 3")
