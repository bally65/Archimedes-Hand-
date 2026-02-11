import time
import os

log_path = "/home/aa598/.openclaw/workspace/robotics/archimedes-hand/5. Deep_LR/training_whole_body.log"

def monitor():
    print("👀 Monitoring Whole-Body Training...")
    last_size = 0
    while True:
        if os.path.exists(log_path):
            current_size = os.path.getsize(log_path)
            if current_size > last_size:
                with open(log_path, 'r') as f:
                    f.seek(last_size)
                    new_content = f.read()
                    if new_content:
                        print(new_content, end='', flush=True)
                last_size = current_size
        time.sleep(5)

if __name__ == "__main__":
    monitor()
