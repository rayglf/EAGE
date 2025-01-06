from pathlib import Path
import torch
ROOT_DIR = Path(__file__).parent
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device=torch.device("cpu")