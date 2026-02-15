import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

log_path = "training_whole_body.log"
if os.path.exists(log_path):
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    rewards = []
    for line in lines:
        if "ep_rew_mean" in line:
            try:
                val = float(line.split("|")[-2].strip())
                rewards.append(val)
            except: pass
            
    if rewards:
        plt.figure(figsize=(10,5))
        plt.plot(rewards, label='Mean Reward')
        plt.title('Archimedes 4.0 Training Progress')
        plt.xlabel('Log Intervals')
        plt.ylabel('Reward')
        plt.grid(True)
        plt.savefig('training_trend.png')
        print("✅ Trend plot saved as training_trend.png")
