import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
import fastapi


print("======================================")
print("       ASTRA-TAD ENVIRONMENT TEST")
print("======================================")

print("Python environment: OK")
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("SciPy:", scipy.__version__)
print("Scikit-learn:", sklearn.__version__)
print("PyTorch:", torch.__version__)
print("FastAPI:", fastapi.__version__)

print("--------------------------------------")
print("CUDA available:", torch.cuda.is_available())
print("--------------------------------------")

print("ASTRA-TAD environment is ready!")