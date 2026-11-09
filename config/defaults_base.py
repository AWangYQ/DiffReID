from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = '0'
# Name of backbone
_C.MODEL.NAME = 'resnet50'
# Last stride of backbone
_C.MODEL.LAST_STRIDE = 1
# Path to pretrained model of backbone
_C.MODEL.PRETRAIN_PATH = ''
# Path to the CLIP checkpoint (e.g. ViT-B-16.pt). If empty or missing, it is downloaded automatically
_C.MODEL.CLIP_PATH = ''

# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' , 'self' , 'finetune'
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'

# If train with BNNeck, options: 'bnneck' or 'no'
_C.MODEL.NECK = 'bnneck'
# If train loss include center loss, options: 'yes' or 'no'. Loss with center loss has different optimizer configuration
_C.MODEL.IF_WITH_CENTER = 'no'

_C.MODEL.ID_LOSS_TYPE = 'softmax'
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0
_C.MODEL.I2T_LOSS_WEIGHT = 1.0

_C.MODEL.METRIC_LOSS_TYPE = 'triplet'
# If train with multi-gpu ddp mode, options: 'True', 'False'
_C.MODEL.DIST_TRAIN = False
# If train with soft triplet loss, options: 'True', 'False'
_C.MODEL.NO_MARGIN = False
# If train with label smooth, options: 'on', 'off'
_C.MODEL.IF_LABELSMOOTH = 'on'
# If train with arcface loss, options: 'True', 'False'
_C.MODEL.COS_LAYER = False

# Transformer setting
_C.MODEL.DROP_PATH = 0.1
_C.MODEL.DROP_OUT = 0.0
_C.MODEL.ATT_DROP_RATE = 0.0
_C.MODEL.TRANSFORMER_TYPE = 'None'
_C.MODEL.STRIDE_SIZE = [16, 16]

# SIE Parameter
_C.MODEL.SIE_COE = 3.0
_C.MODEL.SIE_CAMERA = False
_C.MODEL.SIE_VIEW = False

_C.SD = CN()
_C.SD.FINETUNE_MODE='lora' # lora, all, freeze
_C.SD.UNET = CN()
_C.SD.UNET.PRETRAIN='pretrained/v1-5-pruned.ckpt'
_C.SD.UNET.UPDATE_PROMPT_EMB= True
_C.SD.UNET.LORA_RANK= -1 # # LoRA rank for fine-tuning
                  # it should be assgined in specific config file
_C.SD.UNET.LORA_ALPHA= 1.0
_C.SD.UNET.IN_CHANNELS= 4
_C.SD.UNET.OUT_CHANNELS= 4
_C.SD.UNET.MODEL_CHANNELS= 320
_C.SD.UNET.CONTEXT_DIM= 768
_C.SD.UNET.ATTENTION_RESOLUTIONS= [4, 2, 1]
_C.SD.UNET.DROPOUT = 0.0
_C.SD.UNET.NUM_RES_BLOCKS=2
_C.SD.UNET.CHANNEL_MULT=[1, 2, 4, 4]
_C.SD.UNET.CONV_RESAMPLE= True
_C.SD.UNET.DIMS= 2
_C.SD.UNET.NUM_HEADS=8
_C.SD.UNET.NUM_HEAD_CHANNELS= -1
_C.SD.UNET.NUM_HEADS_UPSAMPLE=-1
_C.SD.UNET.USE_SCALE_SHIFT_NORM=False
_C.SD.UNET.RESBLOCK_UPDOWN=False
_C.SD.UNET.USE_NEW_ATTENTION_ORDER=False
_C.SD.UNET.USE_SPATIAL_TRANSFORMER=True
_C.SD.UNET.TRANSFORMER_DEPTH=1
_C.SD.UNET.USE_CHECKPOINT=True
_C.SD.UNET.LEGACY=False
_C.SD.UNET.LR_MULT= 1.0 # learning rate multiplier on UNet parameters
_C.SD.UNET.CONDITION_BOTTLENECK_TYPE= 'embedding'
_C.SD.UNET.CONDITION_BOTTLENECK_INIT_STD= 1.0
_C.SD.UNET.PROB_TYPE='fc'
_C.SD.UNET.CONDITION_BOTTLENECK_TAU=1.0
_C.SD.UNET.CONDITION_BOTTLENECK_USE_HARD_LABEL=False
_C.SD.DDPM = CN()
_C.SD.DDPM.IMAGE_SIZE= 128
_C.SD.DDPM.TIMESTEPS= 1000
_C.SD.DDPM.MAX_TIMESTEP_SAMPLED= 1000
_C.SD.DDPM.CHANNELS= 4
_C.SD.DDPM.SCALE_FACTOR=0.18215
_C.SD.DDPM.CLIP_DENOISED= True
_C.SD.DDPM.LINEAR_START= 8.5e-4
_C.SD.DDPM.LINEAR_END= 0.012
_C.SD.DDPM.COSINE_S= 8.0e-3
_C.SD.DDPM.GIVEN_BETAS= ''
_C.SD.DDPM.BETA_SCHEDULE= 'linear'
_C.SD.DDPM.ORIGINAL_ELBO_WEIGHT=0.0
_C.SD.DDPM.V_POSTERIOR= 0.0
_C.SD.DDPM.L_SIMPLE_WEIGHT= 1.0
_C.SD.DDPM.PARAMETERIZATION= 'eps'
_C.SD.DDPM.USE_POSITIONAL_ENCODINGS= False
_C.SD.DDPM.LEARN_LOGVAR=False
_C.SD.DDPM.LOGVAR_INIT= 0.0
_C.SD.DDPM.NOISE_ESTIMATE_BEGIN_EP= 1 # Late noise estimation loss
_C.SD.DDPM.AUG_RANDOM_CROP= False # RandomCrop for SD input
_C.SD.AUTOENCODER=CN()
_C.SD.AUTOENCODER.EMBED_DIM= 4
_C.SD.AUTOENCODER.EMBED_DIMZ_CHANNELS= 4
_C.SD.AUTOENCODER.EMBED_DIMRESOLUTION= 256
_C.SD.AUTOENCODER.EMBED_DIMIN_CHANNELS= 3
_C.SD.AUTOENCODER.EMBED_DIMOUT_CH= 3
_C.SD.AUTOENCODER.EMBED_DIMCH= 128
_C.SD.AUTOENCODER.EMBED_DIMCH_MULT= [1,2,4,4]
_C.SD.AUTOENCODER.EMBED_DIMNUM_RES_BLOCKS= 2
_C.SD.AUTOENCODER.EMBED_DIMATTN_RESOLUTIONS= []
_C.SD.AUTOENCODER.EMBED_DIMDROPOUT= 0.0
_C.SD.AUTOENCODER.EMBED_DIMCOLORIZE_NLABELS= 'null'


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [384, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [384, 128]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size
_C.INPUT.PADDING = 10

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('market1501')
# Root directory where datasets should be used (and downloaded if not found)
_C.DATASETS.ROOT_DIR = ('../data')


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 16

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = "Adam"
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 100
# Base learning rate
_C.SOLVER.BASE_LR = 3e-4
# Whether using larger learning rate for fc layer
_C.SOLVER.LARGE_FC_LR = False
# Factor of learning bias
_C.SOLVER.BIAS_LR_FACTOR = 1
# Factor of learning bias
_C.SOLVER.SEED = 1234
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# Margin of triplet loss
_C.SOLVER.MARGIN = 0.3
# Learning rate of SGD to learn the centers of center loss
_C.SOLVER.CENTER_LR = 0.5
# Balanced weight of center loss
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005

# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005

# decay rate of learning rate
_C.SOLVER.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STEPS = (40, 70)
# warm up factor
_C.SOLVER.WARMUP_FACTOR = 0.01
#  warm up epochs
_C.SOLVER.WARMUP_EPOCHS = 5
_C.SOLVER.WARMUP_LR_INIT = 0.01
_C.SOLVER.LR_MIN = 0.000016


_C.SOLVER.WARMUP_ITERS = 500
# method of warm up, option: 'constant','linear'
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.COSINE_MARGIN = 0.5
_C.SOLVER.COSINE_SCALE = 30

# epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = 10
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 100
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = 10
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 128, each GPU will
# contain 16 images per batch
_C.SOLVER.IMS_PER_BATCH = 64

# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #

_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 128
# If test with re-ranking, options: 'True','False'
_C.TEST.RE_RANKING = False
# Path to trained model
_C.TEST.WEIGHT = ""
# Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'
_C.TEST.NECK_FEAT = 'after'
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = 'yes'

# Name for saving the distmat after testing.
_C.TEST.DIST_MAT = "dist_mat.npy"
# Whether calculate the eval score option: 'True', 'False'
_C.TEST.EVAL = False
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = ""
