from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = Path(ROOT_DIR, 'data')
DATA_RAW = Path(DATA_DIR, 'raw')
DATA_TMP = Path(DATA_DIR, 'tmp')
DATA_EXTERNAL = Path(DATA_DIR, 'external')
DATA_PROCESSED = Path(DATA_DIR, 'processed')
TEST_PATH = ROOT_DIR / 'tests'

# Classes
poses = ['cobra', 'corpse', 'lotus', 'mountain', 'tree', 'triangle']
