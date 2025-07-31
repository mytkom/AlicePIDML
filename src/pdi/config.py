from __future__ import annotations

import dataclasses
from typing import List, Literal, Optional

from dataclass_wizard import JSONPyWizard

# TODO: describe those methods here and fill config fields
@dataclasses.dataclass
class OCSVMConfig:
    pass

@dataclasses.dataclass
class IsolationForestConfig:
    pass

@dataclasses.dataclass
class IQRConfig:
    pass

@dataclasses.dataclass
class OutlierFilteringConfig:
    ocsvm: OCSVMConfig = dataclasses.field(default_factory=OCSVMConfig)
    isolation_forest: IsolationForestConfig = dataclasses.field(default_factory=IsolationForestConfig)
    iqr: IQRConfig = dataclasses.field(default_factory=IQRConfig)

# Config related to data preparation process:
# - preprocessing
# - splitting
# - grouping
# It is a commmon config for all data preparation class, desired model
# will choose right data preparation class and use this config to
# initialize it
# TODO: idea, calculate checksum this config class and on the basis of such checksum cache preprocessed datasets in subdirectories
@dataclasses.dataclass
class DataConfig:
    # TODO: describe outlier filtering and options here
    outlier_filtering_methods: OutlierFilteringConfig = dataclasses.field(default_factory=OutlierFilteringConfig)
    outlier_filtering_method: None | Literal["iqr", "ocsvm", "isolation forest"] = None

    # Train/Validation/Test dataset split ratios. Validation is calculated automatically 1 - train_size - test_size
    # 0.64/0.16/0.2 split was proposed in literature: http://arxiv.org/abs/2504.16109
    train_size: float = 0.64
    test_size: float = 0.2

    # Is it data from ALICE Run 3? Different missing values of signals are dependant of this setting.
    is_run_3: bool = True

@dataclasses.dataclass
class AdamWConfig:
    # Weight decay is regularisation method, it prevents overfitting, but with
    # suboptimal value can degrade model performance, use it wisely
    weight_decay: float = 0.0

# TODO: describe and fill parameters
@dataclasses.dataclass
class SGDConfig:
    momentum: float = 0.9
    weight_decay: float = 0.0
    nesterov: bool = True

@dataclasses.dataclass
class OptimizersConfig:
    adamw: AdamWConfig = dataclasses.field(default_factory=AdamWConfig)
    sgd: SGDConfig = dataclasses.field(default_factory=SGDConfig)

@dataclasses.dataclass
class ExponentialLRConfig:
    gamma: float = 0.9

@dataclasses.dataclass
class CosineRestartsLRConfig:
    # Number of epochs of the first cycle
    first_cycle_epochs: int = 20

    # Number of epochs to be increased for every new cycle to the last value
    cycle_epoch_inc: int = 5

@dataclasses.dataclass
class PolynomialLRConfig:
    power: float = 1.0

@dataclasses.dataclass
class ConstantLRConfig:
    factor: float = 1.0
    total_iters: int = 50

@dataclasses.dataclass
class LRSchedulersConfig:
    exponential: ExponentialLRConfig = dataclasses.field(default_factory=ExponentialLRConfig)
    cosine_restarts: CosineRestartsLRConfig = dataclasses.field(default_factory=CosineRestartsLRConfig)
    polynomial: PolynomialLRConfig = dataclasses.field(default_factory=PolynomialLRConfig)
    constant: ConstantLRConfig = dataclasses.field(default_factory=ConstantLRConfig)

@dataclasses.dataclass
class ValidationConfig:
    batch_size: int = 8192

    num_workers: int = 8

    # Evaluate training results after this count of training epochs
    validate_every: int = 5


@dataclasses.dataclass
class TrainingConfig:
    # Choose optimizer
    optimizers: OptimizersConfig = dataclasses.field(default_factory=OptimizersConfig)
    optimizer: Literal["adamw", "sgd"] = "adamw"

    # Choose learning rate scheduler
    lr_schedulers: LRSchedulersConfig = dataclasses.field(default_factory=LRSchedulersConfig)
    lr_scheduler: Literal["exponential", "cosine_restarts", "polynomial", "constant"] | None = "exponential"

    # Loss (risk) function to be minimized
    loss: Literal["cross entropy"] = "cross entropy"

    # Choose start learning rate
    start_lr: float = 0.003

    # Choose device for training (cuda is Nvidia GPU)
    device: Literal["cuda", "cpu"] = "cuda"

    # How many training steps needs to be done before logging results
    steps_to_log: int = 50

    # Batch size of observations, update of model weights will be done after full batch.
    # Bigger batch size makes direction of optimization descent more stable.
    # Bigger batch size utilizes GPU more.
    batch_size: int = 512

    # If early stopping criterion would not be met, then max_epochs of training will be done
    max_epochs: int = 50

    # How many epochs without progress needs to be done before early stopping
    early_stopping_epoch_count: int = 5

    # Defines what is progress in loss minimization; if (1 - current_loss / min_loss > threshold),
    # then epochs without progress counter is being reset
    early_stopping_progress_threshold: float = .001

    # Number of subprocesses (or threads idk) pre-loading batches in parallel and delivering it
    # to main training process. It can speed up training process with drawback of bigger CPU
    # and RAM utilization.
    num_workers: int = 4

    # If to undersample observations by particle types, e.g. if training model for proton classification,
    # ~ 3% od data is proton observations, and 97% of data is not proton observations, if this option is set
    # to true, loss function will be weighted accordingly to this ratio
    weight_particles_species: bool = False

    # If to undersample observations by missing detectors groups on the DataLoader stage.
    # For majority classes it randomly selects different subsample every epoch.
    # Status: It proved to be necessary on Run3 data, because of 70% TPC only observations
    undersample_missing_detectors: bool = True

    # TODO: description
    undersample_pions: bool = True


def mlp_default_hidden_layers():
    return [64, 32, 16]

@dataclasses.dataclass
class MLPConfig:
    # List of neurons in layer dimensions, first is input layer, then hidden layers, at the end output layer
    hidden_layers: List[int] = dataclasses.field(default_factory=mlp_default_hidden_layers)

    # delete is equivalent complete-only data, mean fills missing cells with mean of this column
    # on the whole dataset and linear regression fills missing values by linear regression over
    # filled column
    missing_data_strategy: Literal["mean", "linear regression"] = "mean"

    # Choose non-linear activation function to be used after each layer
    activation: Literal["ReLU"] = "ReLU"

    # Dropout, regularisation term for Neural Network
    dropout: float = 0.9

@dataclasses.dataclass
class EnsembleConfig:
    # ids of groups of missing detectors
    group_ids: List[int] = dataclasses.field(default_factory=list)

    # List of neurons in layer dimensions of hidden layers, input and output is fixed
    hidden_layers: List[int] = dataclasses.field(default_factory=mlp_default_hidden_layers)

    # Choose non-linear activation function to be used after each layer
    activation: Literal["ReLU"] = "ReLU"

    # Dropout, regularisation term for Neural Network
    dropout: float = 0.9

@dataclasses.dataclass
class AttentionConfig:
    # Embedding dimension and hidden dimension for MLP doing preliminary embedding settings
    embed_hidden: int = 128
    embed_dim: int = 32

    # dimension of feed forward (MLP) neural network hidden layer
    ff_hidden: int = 128

    # Pooling dimension of AttentionPulling module (see models.py for more details)
    pool_hidden: int = 64

    # Number of heads in multi-head attention
    num_heads: int = 2

    # Number of blocks in each head
    num_blocks: int = 2

    # Non-linear activation function for feed-forward network
    activation: Literal["ReLU"] = "ReLU"

    # Dropout for feed-forward network - regularisation
    dropout: float = 0.4

@dataclasses.dataclass
class AttentionDANNConfig:
    # list of hidden layers sizes (number of neurons in each layer) for domain classifier
    dom_hidden_layers: List[int] = dataclasses.field(default_factory=mlp_default_hidden_layers)

    # Standard Attention model configuration
    attention: AttentionConfig = dataclasses.field(default_factory=AttentionConfig)

    # Multiplier of negative loss of domain classifier used in backpropagation algorithm
    # the bigger, the stronger Domain Adaptation effect, if too high it can degrade class label
    # classification performance, while not giving much better transformed feature space
    # in terms of source/target domain indistinguishibility
    alpha: float = 2.0


@dataclasses.dataclass
class ModelConfig:
    # Choose architecture of Deep Neural Network to use
    # - mlp: single multi-layer perceptron, which cannot handle missing data --- it needs to be filled (e.g. by mean)
    # - ensemble: ensemble of MLPs, one for each missing detector combination (4 combinations)
    # - attention: attention-based neural network, it can handle missing data by one-hot encoding of input
    # - attention_dann: attention-based neural network with domain adversarial neural network approach (domain classifier added)
    architecture: Literal["mlp", "ensemble", "attention", "attention_dann"] = "attention"
    mlp: MLPConfig = dataclasses.field(default_factory=MLPConfig)
    ensemble: EnsembleConfig = dataclasses.field(default_factory=EnsembleConfig)
    attention: AttentionConfig = dataclasses.field(default_factory=AttentionConfig)
    attention_dann: AttentionDANNConfig = dataclasses.field(default_factory=AttentionDANNConfig)

    # If you want to start training from some checkpoint, you can pass the path to the
    # directory with best.pt weights and metadata.json files of the model. Make sure to
    # set correct architecture for your weights.
    pretrained_model_dirpath: Optional[str] = None

@dataclasses.dataclass
class SweepConfig:
    # Filepath to the wandb sweep config
    config: str = ""

    # Name for the sweep for WandB
    name: str = "sweep"

    # Name of the project in WandB
    project_name: str = "default_project"

@dataclasses.dataclass
class Config(JSONPyWizard):
    # data preprocessing, loading and balancing methods
    data: DataConfig = dataclasses.field(default_factory=DataConfig)

    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)

    sweep: SweepConfig = dataclasses.field(default_factory=SweepConfig)

    training: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)

    validation: ValidationConfig = dataclasses.field(default_factory=ValidationConfig)

    # Paths to simulated datasets obtained using O2Physics's PIDMLProducer with ML option enabled
    sim_dataset_paths: List[str] = dataclasses.field(default_factory=list)

    # Paths to experimental datasets obtained using O2Physics's PIDMLProducer with Data option enabled
    # Important: It will be used only by Domain Adaptation models, if you train using just
    # simulated data, then probably, you do not need it
    exp_dataset_paths: List[str] = dataclasses.field(default_factory=list)
    project_dir: str = "project"
    log_dir: str = "logs"
    # TODO: check if can be easily adopted, it can give us performance boost
    mixed_precision: str = "no"
    seed: int = 0
    config_path: Optional[str] = None

@dataclasses.dataclass
class OneParticleConfig:
    config: Optional[str] = None
    particle: Literal["pion", "kaon", "proton", "antipion", "antikaon", "antiproton"] = "pion"

@dataclasses.dataclass
class AllParticlesConfig:
    # Default config loaded, when no particle specific config is provided.
    # It can be used for example to test all particle species training on single config
    # or to do so with most of the species, but overwrite e.g. pions.
    all: Optional[str] = None

    pion: Optional[str] = None
    kaon: Optional[str] = None
    proton: Optional[str] = None
    antipion: Optional[str] = None
    antikaon: Optional[str] = None
    antiproton: Optional[str] = None


