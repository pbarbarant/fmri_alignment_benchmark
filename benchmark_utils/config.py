from pathlib import Path
from socket import gethostname

if gethostname().startswith("drago"):  # drago* machine
    root = Path("/storage/store2/work/dfouchar/fmralign_benchopt_data/")
    # path to memory
    MEMORY = "/storage/store2/work/dfouchar/tmp"

elif gethostname().startswith("mar"):  # margaret machine
    root = Path("/data/parietal/store2/work/dfouchar/fmralign_benchopt_data/")
    # path to memory
    MEMORY = "/data/parietal/store2/work/dfouchar/tmp"

else:  # local machine
    root = Path("/Users/df/datasets")
    # path to memory
    MEMORY = "/Users/df/tmp"

# path to IBC RSVP data
DATA_PATH_IBC_RSVP = root / "IBC_RSVP"

# path to IBC Sound data
DATA_PATH_IBC_SOUND = root / "IBC_Sound"

# path to IBC Mathlang data
DATA_PATH_IBC_MATHLANG_AUDIO = root / "IBC_MathLangAudio"

# path to Neuromod data
DATA_PATH_NEUROMOD = root / "Neuromod"

# path to BOLD5000 data
DATA_PATH_BOLD5000 = root / "BOLD5000"
