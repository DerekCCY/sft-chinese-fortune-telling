import os

print("Starting Full Ziwei Chart Pipeline...")

os.system("python -m data_generation.gen_births --count 500")
os.system("python -m data_generation.gen_charts")

print("Pipeline Finished.")
