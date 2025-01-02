from yacs.config import CfgNode

_C = CfgNode()
_C.TRAIN_DIR = 'WaterDropSample/train'
_C.VAL_DIR = 'WaterDropSample/valid'
_C.N_HIS = 7
_C.ROLLOUT_STEPS = 1001
_C.PRED_STEPS = 1
_C.MAX_VAL = 2
_C.KINEMATIC_PARTICLE_ID = 3
_C.NUM_PARTICLE_TYPES = 9

_C.SOLVER = CfgNode()
_C.SOLVER.BASE_LR = 0.0001
_C.SOLVER.MIN_LR = 0.000001
_C.SOLVER.VAL_INTERVAL = 50
_C.SOLVER.WEIGHT_DECAY = 0.0
_C.SOLVER.WARMUP_ITERS = 0
_C.SOLVER.MAX_ITERS = 5e3
_C.SOLVER.BATCH_SIZE = 1
_C.SOLVER.LR_DECAY_INTERVAL = 5e3

_C.NET = CfgNode()
_C.NET.RADIUS = 0.015
_C.NET.NOISE = 0.00067
_C.NET.PARTICLE_EMB_SIZE = 16
_C.NET.MAX_EDGE_PER_PARTICLE = 150
_C.NET.SELF_EDGE = True
_C.NET.NODE_FEAT_DIM_IN = 32
_C.NET.EDGE_FEAT_DIM_IN = 3
_C.NET.GNN_LAYER = 10
_C.NET.HIDDEN_SIZE = 128
_C.NET.OUT_SIZE = 2
_C.NET.recover_what = 'acc' # ['acc', vel', 'pos']