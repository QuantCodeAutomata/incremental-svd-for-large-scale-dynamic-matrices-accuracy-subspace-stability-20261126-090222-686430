"""Paper-derived parameters for every experiment.

Source: Staykov, S. (2026), "Incremental SVD for Large-Scale Dynamic
Matrices: Accuracy, Subspace Stability, Refresh Strategies, and Financial
Factor-Based Risk Models", arXiv:2605.24514. Section numbers below refer to
this paper. Where a value is not stated in the paper text, the comment says
so and the value follows the authors' reference implementation
(https://github.com/stilyanst/iSVD), which produced the paper's reported
numbers.
"""

# ---------------------------------------------------------------------------
# Shared synthetic matrix generator (Secs. 4.1-4.2, 4.6)
# ---------------------------------------------------------------------------
RANK1_ROWS = 50          # m, rows of the synthetic matrix (Sec. 4.1)
RANK1_COLS = 40          # n, columns of the synthetic matrix (Sec. 4.1)
RANK1_TRUE_RANK = 5      # true latent rank of A (Sec. 4.1)
RANK1_DELTA_STD = 0.05   # std of the rank-1 entry increment delta ~ N(0, 0.05^2) (Sec. 4.1)

# ---------------------------------------------------------------------------
# Experiment 1: rank-1 streaming, tracking and drift under periodic refresh (Sec. 4.1)
# ---------------------------------------------------------------------------
EXP1_SEED = 0                       # random seed (not specified in the paper; fixed for reproducibility)
EXP1_N_STEPS = 10_000                # number of rank-1 updates (Sec. 4.1)
EXP1_TRUNCATION_RANK = 5             # k, matches the true rank (Sec. 4.1)
EXP1_REFRESH_PERIODS = (None, 1000, 5000)  # no-refresh, refresh every 1000, every 5000 steps (Sec. 4.1)
EXP1_LOG_EVERY = 20                  # steps between oracle full-SVD evaluations (not specified in the paper; dense enough to show the sawtooth)

# ---------------------------------------------------------------------------
# Experiment 2: effect of truncation rank k on error ratio and tracking (Sec. 4.2)
# ---------------------------------------------------------------------------
EXP2_SEED = 0
EXP2_N_STEPS = 6_000                 # number of rank-1 updates (Sec. 4.2)
EXP2_RANKS = (5, 8, 12)              # truncation ranks k tested (Sec. 4.2)
EXP2_REFRESH_EVERY = 2_000           # refresh period (Sec. 4.2)
EXP2_LOG_EVERY = 20
EXP2_TIMING_RUNS = 5                 # median over 5 runs for the per-update time (Sec. 4.2, Fig. 3)
EXP2_TIMING_STEPS = 300              # updates timed per run (consistent with Sec. 4.6 runtime protocol)

# ---------------------------------------------------------------------------
# Experiment 3: structural growth, row and column appends (Sec. 4.3)
# ---------------------------------------------------------------------------
EXP3_SEED = 0
EXP3_INIT_ROWS = 50                  # initial number of rows (not specified in the paper; kept consistent with Secs. 4.1-4.2)
EXP3_INIT_COLS = 40                  # initial number of columns (not specified in the paper)
EXP3_TRUE_RANK = 6                   # true rank of the generating model (Sec. 4.3)
EXP3_NOISE_STD = 0.05                # additive Gaussian noise std (not specified in the paper; set equal to the Sec. 4.1 update scale)
EXP3_WORKING_RANK = 4                # tracked truncation rank k, below the true rank 6 (Sec. 4.3)
EXP3_N_APPENDS = 200                 # number of row/column appends (Sec. 4.3)
EXP3_REFRESH_EVERY = 50              # refresh period (Sec. 4.3)
EXP3_LOG_EVERY = 1                   # every append is logged (Sec. 4.3, Fig. 4 shows per-step detail)

# ---------------------------------------------------------------------------
# Experiment 4: mixed streaming, rank-1 updates and row appends (Sec. 4.4)
# ---------------------------------------------------------------------------
EXP4_SEED = 0
EXP4_ROWS = RANK1_ROWS                # 50x40 rank-5 matrix, same type as Sec. 4.1 (Sec. 4.4)
EXP4_COLS = RANK1_COLS
EXP4_TRUE_RANK = RANK1_TRUE_RANK
EXP4_TRUNCATION_RANK = 5               # rank-5 SVD initialization, per the experiment methodology
EXP4_N_STEPS = 2_000                  # total stream length (Sec. 4.4)
EXP4_ROW_PERIOD = 50                  # one new row every 50 steps (Sec. 4.4)
EXP4_DELTA_STD = RANK1_DELTA_STD       # entry-update increment scale (Sec. 4.4)
EXP4_ROW_NOISE_STD = 0.05              # additive noise on new rows (not specified in the paper; set equal to the entry-update scale)
EXP4_REFRESH_EVERY = 200               # refresh period (Sec. 4.4)
EXP4_LOG_EVERY = 5

# ---------------------------------------------------------------------------
# Experiment 5: a posteriori error bound and bound-triggered refresh (Sec. 4.5)
# ---------------------------------------------------------------------------
EXP5_SEEDS = (0, 1, 2, 3, 4)          # five random seeds, seed 0 is the Sec. 4.1 stream (Sec. 4.5)
EXP5_N_STEPS = 10_000
EXP5_TRUNCATION_RANK = 5
EXP5_PERIODIC_REFRESH = 1000           # T_refresh = 1000 (Sec. 4.5)
EXP5_BOUND_TAUS = (0.4, 0.2, 0.1, 0.05)  # bound-triggered tolerances tau (Sec. 4.5, eq. 16)
EXP5_LOG_EVERY = 20
EXP5_TOL = 1e-10                       # residual-norm tolerance (Sec. 2, numerical tolerance)

# ---------------------------------------------------------------------------
# Experiment 6: runtime scaling with matrix size and truncation rank (Sec. 4.6/4.2)
# ---------------------------------------------------------------------------
EXP6_SEED = 0
EXP6_SMALL_SHAPE = (50, 40)            # small matrix (Fig. 3a)
EXP6_SMALL_RANKS = (5, 8, 12)
EXP6_LARGE_SHAPE = (500, 400)           # larger matrix (Fig. 3b)
EXP6_LARGE_RANKS = (5, 10, 20, 40)
EXP6_TRUE_RANK = 5
EXP6_N_UPDATES = 300                    # rank-1 updates timed per k (Sec. 4.2/4.6)
EXP6_TIMING_RUNS = 5                    # median over 5 runs (Sec. 4.2/4.6)

# ---------------------------------------------------------------------------
# Experiment 7: online factor model on ETF returns (Sec. 5)
# ---------------------------------------------------------------------------
# ETF universe: the 60 tickers of the paper's Section 5 panel, taken from the
# authors' data-preparation script experiments/prepare_etf_data.py (the paper
# text does not print the ticker list, Sec. 5).
EXP7_TICKERS = [
    # Broad US equity / style (authors' experiments/prepare_etf_data.py)
    "SPY", "IVV", "VTI", "QQQ", "DIA", "IWM", "IWN", "IWO", "VTV", "VUG",
    # International / regional equity
    "VEA", "VWO", "EFA", "EEM", "EWJ", "EWU", "EWG", "EWQ", "EWL", "EWA",
    "EWC", "EWH", "EWY", "EWZ", "EWW", "EWD",
    # US sector ETFs (GICS sectors)
    "XLF", "XLK", "XLE", "XLY", "XLP", "XLV", "XLI", "XLB", "XLRE", "XLU", "XLC",
    # Fixed income / rates / credit
    "TLT", "IEF", "SHY", "BND", "AGG", "LQD", "HYG", "JNK", "TIP", "EMB",
    # Commodities / real assets / thematics
    "GLD", "SLV", "GDX", "GDXJ", "USO", "DBC", "DBA", "VNQ", "IYR", "XME",
    "XOP", "KRE", "SMH",
]
EXP7_START_DATE = "2010-01-01"          # download window start (Sec. 5)
EXP7_END_DATE = "2024-12-31"            # download window end (Sec. 5)
EXP7_INITIAL_WINDOW = 250               # T0, initial full-SVD window in trading days (Sec. 5)
EXP7_PRIMARY_RANK = 5                   # primary truncation rank k (Sec. 5)
EXP7_TIMESERIES_REFRESH = (None, 100, 20)  # no_refresh, refresh_100, refresh_20 (Sec. 5)
EXP7_GRID_RANKS = (3, 5, 8, 12, 20)     # rank x refresh grid (Sec. 5.3)
EXP7_GRID_REFRESH = (None, 30, 50, 100)
EXP7_BOUND_RANKS = (5, 8, 12)           # ranks for bound-triggered refresh (Sec. 5.3)
EXP7_BOUND_TAUS = (0.5, 0.6, 0.7)       # bound tolerances tau (Sec. 5.3, eq. 16)
EXP7_EVAL_EVERY = 5                     # trading days between metric evaluations (paper says "every few days"; not precisely specified)
EXP7_N_RANDOM_PORTFOLIOS = 10           # random long-short portfolios (Sec. 5.2)
EXP7_PORTFOLIO_SEED = 0                 # seed for the random portfolios (not specified in the paper)
# The universe is downloaded in full: 60 daily series over ~15 years is a few
# minutes through the QuantX Dataloader, well inside the available time.
EXP7_UNIVERSE_SIZE_PAPER = 60
EXP7_UNIVERSE_SIZE_RUN = 60
