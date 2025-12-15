from config.config import load_config
cfg = load_config()
W = cfg.kernel.weights()
EPS = cfg.kernel.eps
PERIOD = cfg.kernel.structure_period

