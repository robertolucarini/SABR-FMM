import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

DATA_DIR = "data"
LMM_TENOR = 1.0
CALIB_TENOR = 1.0 
CORR_MODE = 'two_param' # exp, pca, two_param
CHECK_CALIBRATION = True

# market assumptions
RHO_OIS_EUR = 0.90 # Pallavicini
DECAY_B = 0.05
RHO_INF = 0.03 
PCA_FACTORS = 3.0

# bermudan 
EXERCISE_DATES = [1.0, 2.0, 3.0, 4.0, 5.0]
STRIKE = 0.05

# greeks
BUMP_DELTA = 0.0001 # delta bump
BUMP_VOL = 0.0001 # vega bump

# simulation
RNG_TYPE = 'sobol' # sobol, standard
ANTITHETIC_VAR = True
B_BRIDGE = True
PREDICT_CORRECT = True
SIM_STEPS = 50 # time steps
SIM_PATHS = 2**13 # MC paths
SEED = 56 
