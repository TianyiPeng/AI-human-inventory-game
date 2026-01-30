"""
Re-run llm_to_or strategy for specific instances that failed.
Parallel execution for faster processing.
"""

import os
import sys
import subprocess
import json
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Incomplete instances from batch_log_20260128_193052.txt
INCOMPLETE_INSTANCES = [
    "156231001",
    "111586001", 
    "160442043",
    "160442010",
    "160442007",
    "179123001",
    "179950001",
    "189616006",
    "189634001",
    "189616008",
    "224606019",
    "243937001",
    "253448003",
    "253448001",
    "294008002",
]

BASE_DIR = r"D:\TextArena\examples\benchmark\real_small_batch\lead_time_0"
MODEL = "google/gemini-3-pro-preview"
SCRIPT_PATH = r"D:\TextArena\examples\llm_to_or_csv_demo.py"
PARALLEL_WORKERS = 5  # Number of parallel workers


def run_llm_to_or(instance_name: str) -> tuple:
    """Run llm_to_or for a single instance. Returns (instance_name, reward, error)."""
    instance_dir = os.path.join(BASE_DIR, instance_name)
    test_csv = os.path.join(instance_dir, "test.csv")
    train_csv = os.path.join(instance_dir, "train.csv")
    log_path = os.path.join(instance_dir, "llm_to_or_1.txt")
    
    if not os.path.exists(test_csv):
        return instance_name, None, f"test.csv not found"
    
    cmd = [
        sys.executable,
        SCRIPT_PATH,
        "--demand-file", test_csv,
        "--real-instance-train", train_csv,
        "--promised-lead-time", "0",
        "--model", MODEL,
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=2400,  # 40 minutes
            cwd=r"D:\TextArena"
        )
        
        output = result.stdout + result.stderr
        
        # Save log
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"[Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n")
            f.write(f"[Script: llm_to_or (rerun)]\n")
            f.write(f"[Instance: {instance_dir}]\n\n")
            f.write(output)
        
        # Extract reward
        import re
        reward_match = re.search(r">>>\s*Total Reward[^:]*:\s*\$(-?[\d,]+\.?\d*)\s*<<<", output)
        if reward_match:
            reward = float(reward_match.group(1).replace(',', ''))
            return instance_name, reward, None
        
        return instance_name, None, f"Could not extract reward (exit code: {result.returncode})"
        
    except subprocess.TimeoutExpired:
        return instance_name, None, "Timeout after 40 minutes"
    except Exception as e:
        return instance_name, None, str(e)


def update_benchmark_results(instance_name: str, reward: float):
    """Update benchmark_results.json with the new llm_to_or result."""
    instance_dir = os.path.join(BASE_DIR, instance_name)
    results_path = os.path.join(instance_dir, "benchmark_results.json")
    
    if not os.path.exists(results_path):
        print(f"Warning: {results_path} not found")
        return
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    # Update llm_to_or results
    data['results']['llm_to_or'] = {
        'rewards': [reward],
        'mean': reward,
        'std': 0.0,
        'min': reward,
        'max': reward,
        'count': 1,
        'ratio_to_perfect': reward / data['summary']['perfect_score'] if data['summary']['perfect_score'] else None
    }
    
    # Update summary ratios
    if data['summary']['perfect_score']:
        data['summary']['ratios']['llm_to_or'] = reward / data['summary']['perfect_score']
    
    # Clear errors for llm_to_or
    if 'errors' in data and 'llm_to_or' in data.get('errors', {}):
        del data['errors']['llm_to_or']
    if data.get('errors') == {}:
        data['errors'] = {}
    
    with open(results_path, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(BASE_DIR, f"rerun_llm_to_or_{timestamp}.txt")
    
    results = []
    
    def log(msg):
        timestamped = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(timestamped)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(timestamped + "\n")
    
    log(f"Re-running llm_to_or for {len(INCOMPLETE_INSTANCES)} instances")
    log(f"Model: {MODEL}")
    log(f"Parallel workers: {PARALLEL_WORKERS}")
    log(f"Log: {log_path}")
    log("")
    
    # Run in parallel
    with ProcessPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        future_to_instance = {
            executor.submit(run_llm_to_or, name): name 
            for name in INCOMPLETE_INSTANCES
        }
        
        completed = 0
        total = len(INCOMPLETE_INSTANCES)
        
        for future in as_completed(future_to_instance):
            completed += 1
            instance_name, reward, error = future.result()
            
            if reward is not None:
                log(f"[{completed}/{total}] SUCCESS: {instance_name} -> ${reward:.2f}")
                update_benchmark_results(instance_name, reward)
                results.append((instance_name, "SUCCESS", reward))
            else:
                log(f"[{completed}/{total}] FAILED: {instance_name} -> {error}")
                results.append((instance_name, "FAILED", error))
    
    # Summary
    log("")
    log("=" * 60)
    log("SUMMARY")
    log("=" * 60)
    
    success_count = sum(1 for _, status, _ in results if status == "SUCCESS")
    log(f"Success: {success_count}/{len(results)}")
    log("")
    
    for name, status, detail in sorted(results):
        if status == "SUCCESS":
            log(f"  {name}: ${detail:.2f}")
        else:
            log(f"  {name}: FAILED - {detail}")
    
    log("")
    log(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
