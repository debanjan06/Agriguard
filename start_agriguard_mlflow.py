
import subprocess
import os
import sys

# Change to AgriGuard directory
os.chdir(r"C:\Users\DEBANJAN SHIL\Documents\AgriGuard")

# Start MLflow UI with AgriGuard data only
cmd = [
    sys.executable, "-m", "mlflow", "ui",
    "--backend-store-uri", r"C:\Users\DEBANJAN SHIL\Documents\AgriGuard\agriguard_mlruns",
    "--host", "127.0.0.1", 
    "--port", "5001"
]

print("Starting MLflow UI for AgriGuard only...")
print(f"Command: {' '.join(cmd)}")
print("Navigate to: http://localhost:5001")

subprocess.run(cmd)
