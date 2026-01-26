"""
Generate 200 synthetic demand trajectories for benchmark.
"""

import argparse
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union
import numpy as np

DistConfig = Dict[str, Any]
Segment = Tuple[int, int, DistConfig]

def make_normal(mu, sigma):
    return {'type': 'normal', 'mu': mu, 'sigma': sigma}

def make_uniform(a, b):
    return {'type': 'uniform', 'a': a, 'b': b}

def make_ar1(phi, c, sigma):
    return {'type': 'ar1', 'phi': phi, 'c': c, 'sigma': sigma}

def make_changepoint(segments):
    return {'type': 'changepoint', 'segments': segments}

DISTRIBUTIONS = {}

# P01 Stationary
DISTRIBUTIONS[('p01_stationary_iid', 'v1_normal_100_25')] = {
    'pattern_type': 'stationary',
    'config': make_normal(100, 25),
    'description': 'N(100, 25)'
}
DISTRIBUTIONS[('p01_stationary_iid', 'v2_normal_100_40')] = {
    'pattern_type': 'stationary',
    'config': make_normal(100, 40),
    'description': 'N(100, 40)'
}
DISTRIBUTIONS[('p01_stationary_iid', 'v3_normal_100_15')] = {
    'pattern_type': 'stationary',
    'config': make_normal(100, 15),
    'description': 'N(100, 15)'
}
DISTRIBUTIONS[('p01_stationary_iid', 'v4_uniform_50_150')] = {
    'pattern_type': 'stationary',
    'config': make_uniform(50, 150),
    'description': 'Uniform[50, 150]'
}

# P02 Mean Increase
DISTRIBUTIONS[('p02_mean_increase', 'v1_100to200')] = {
    'pattern_type': 'changepoint',
    'config': make_changepoint([(1, 15, make_normal(100, 25)), (16, 50, make_normal(200, 35))]),
    'description': 'N(100,25)->N(200,35)'
}
DISTRIBUTIONS[('p02_mean_increase', 'v2_100to150')] = {
    'pattern_type': 'changepoint',
    'config': make_changepoint([(1, 15, make_normal(100, 25)), (16, 50, make_normal(150, 30))]),
    'description': 'N(100,25)->N(150,30)'
}
DISTRIBUTIONS[('p02_mean_increase', 'v3_100to300')] = {
    'pattern_type': 'changepoint',
    'config': make_changepoint([(1, 15, make_normal(100, 25)), (16, 50, make_normal(300, 50))]),
    'description': 'N(100,25)->N(300,50)'
}
DISTRIBUTIONS[('p02_mean_increase', 'v4_100to200_samevar')] = {
    'pattern_type': 'changepoint',
    'config': make_changepoint([(1, 15, make_normal(100, 25)), (16, 50, make_normal(200, 25))]),
    'description': 'N(100,25)->N(200,25)'
}

# P03 Mean Decrease
DISTRIBUTIONS[('p03_mean_decrease', 'v1_100to50')] = {
    'pattern_type': 'changepoint',
    'config': make_changepoint([(1, 15, make_normal(100, 25)), (16, 50, make_normal(50, 18))]),
    'description': 'N(100,25)->N(50,18)'
}
DISTRIBUTIONS[('p03_mean_decrease', 'v2_100to70')] = {
    'pattern_type': 'changepoint',
    'config': make_changepoint([(1, 15, make_normal(100, 25)), (16, 50, make_normal(70, 20))]),
    'description': 'N(100,25)->N(70,20)'
}
DISTRIBUTIONS[('p03_mean_decrease', 'v3_100to30')] = {
    'pattern_type': 'changepoint',
    'config': make_changepoint([(1, 15, make_normal(100, 25)), (16, 50, make_normal(30, 15))]),
    'description': 'N(100,25)->N(30,15)'
}
DISTRIBUTIONS[('p03_mean_decrease', 'v4_150to80')] = {
    'pattern_type': 'changepoint',
    'config': make_changepoint([(1, 15, make_normal(150, 30)), (16, 50, make_normal(80, 22))]),
    'description': 'N(150,30)->N(80,22)'
}

# P04 Increasing Trend
DISTRIBUTIONS[('p04_increasing_trend', 'v1_linear_100t')] = {
    'pattern_type': 'trend',
    'config': make_normal(lambda t: 100 * t, lambda t: 25 * math.sqrt(t)),
    'description': 'N(100t, 25*sqrt(t))'
}
DISTRIBUTIONS[('p04_increasing_trend', 'v2_linear_50_3t')] = {
    'pattern_type': 'trend',
    'config': make_normal(lambda t: 50 + 3 * t, 20),
    'description': 'N(50+3t, 20)'
}
DISTRIBUTIONS[('p04_increasing_trend', 'v3_exp_1_05')] = {
    'pattern_type': 'trend',
    'config': make_normal(lambda t: 100 * (1.05 ** t), 25),
    'description': 'N(100*1.05^t, 25)'
}
DISTRIBUTIONS[('p04_increasing_trend', 'v4_linear_100_2t')] = {
    'pattern_type': 'trend',
    'config': make_normal(lambda t: 100 + 2 * t, lambda t: 25 * math.sqrt(t)),
    'description': 'N(100+2t, 25*sqrt(t))'
}

# P05 Decreasing Trend
DISTRIBUTIONS[('p05_decreasing_trend', 'v1_200_minus_3t')] = {
    'pattern_type': 'trend',
    'config': make_normal(lambda t: max(200 - 3 * t, 50), 25),
    'description': 'N(max(200-3t, 50), 25)'
}
DISTRIBUTIONS[('p05_decreasing_trend', 'v2_exp_decay_0_97')] = {
    'pattern_type': 'trend',
    'config': make_normal(lambda t: 200 * (0.97 ** t), 20),
    'description': 'N(200*0.97^t, 20)'
}
DISTRIBUTIONS[('p05_decreasing_trend', 'v3_150_minus_2t')] = {
    'pattern_type': 'trend',
    'config': make_normal(lambda t: max(150 - 2 * t, 30), 20),
    'description': 'N(max(150-2t, 30), 20)'
}
DISTRIBUTIONS[('p05_decreasing_trend', 'v4_200_div_sqrt_t')] = {
    'pattern_type': 'trend',
    'config': make_normal(lambda t: 200 / math.sqrt(t), 15),
    'description': 'N(200/sqrt(t), 15)'
}

# P06 Variance Change
DISTRIBUTIONS[('p06_variance_change', 'v1_normal_to_uniform')] = {
    'pattern_type': 'changepoint',
    'config': make_changepoint([(1, 15, make_normal(100, 25)), (16, 50, make_uniform(0, 200))]),
    'description': 'N(100,25)->Uniform[0,200]'
}
DISTRIBUTIONS[('p06_variance_change', 'v2_var_increase')] = {
    'pattern_type': 'changepoint',
    'config': make_changepoint([(1, 15, make_normal(100, 25)), (16, 50, make_normal(100, 50))]),
    'description': 'N(100,25)->N(100,50)'
}
DISTRIBUTIONS[('p06_variance_change', 'v3_var_decrease')] = {
    'pattern_type': 'changepoint',
    'config': make_changepoint([(1, 15, make_normal(100, 50)), (16, 50, make_normal(100, 20))]),
    'description': 'N(100,50)->N(100,20)'
}
DISTRIBUTIONS[('p06_variance_change', 'v4_uniform_to_normal')] = {
    'pattern_type': 'changepoint',
    'config': make_changepoint([(1, 15, make_uniform(50, 150)), (16, 50, make_normal(100, 15))]),
    'description': 'Uniform[50,150]->N(100,15)'
}

# P07 Seasonal
DISTRIBUTIONS[('p07_seasonal', 'v1_period10_amp30')] = {
    'pattern_type': 'seasonal',
    'config': make_normal(lambda t: 100 + 30 * math.sin(2 * math.pi * t / 10), 25),
    'description': 'N(100+30*sin, 25) T=10'
}
DISTRIBUTIONS[('p07_seasonal', 'v2_period5_amp50')] = {
    'pattern_type': 'seasonal',
    'config': make_normal(lambda t: 100 + 50 * math.sin(2 * math.pi * t / 5), 25),
    'description': 'N(100+50*sin, 25) T=5'
}
DISTRIBUTIONS[('p07_seasonal', 'v3_period25_amp40')] = {
    'pattern_type': 'seasonal',
    'config': make_normal(lambda t: 100 + 40 * math.sin(2 * math.pi * t / 25), 25),
    'description': 'N(100+40*sin, 25) T=25'
}
DISTRIBUTIONS[('p07_seasonal', 'v4_multiplicative')] = {
    'pattern_type': 'seasonal',
    'config': make_normal(lambda t: 100 * (1 + 0.3 * math.sin(2 * math.pi * t / 10)), 25),
    'description': 'N(100*(1+0.3*sin), 25)'
}

# P08 Multi Changepoint
DISTRIBUTIONS[('p08_multi_changepoint', 'v1_up_then_down')] = {
    'pattern_type': 'multi_changepoint',
    'config': make_changepoint([(1, 15, make_normal(100, 25)), (16, 35, make_normal(150, 30)), (36, 50, make_normal(80, 20))]),
    'description': '100->150->80'
}
DISTRIBUTIONS[('p08_multi_changepoint', 'v2_down_then_up')] = {
    'pattern_type': 'multi_changepoint',
    'config': make_changepoint([(1, 15, make_normal(100, 25)), (16, 35, make_normal(60, 20)), (36, 50, make_normal(140, 30))]),
    'description': '100->60->140'
}
DISTRIBUTIONS[('p08_multi_changepoint', 'v3_var_high_then_low')] = {
    'pattern_type': 'multi_changepoint',
    'config': make_changepoint([(1, 15, make_normal(100, 25)), (16, 35, make_normal(100, 50)), (36, 50, make_normal(100, 20))]),
    'description': 'var:25->50->20'
}
DISTRIBUTIONS[('p08_multi_changepoint', 'v4_mild_fluctuations')] = {
    'pattern_type': 'multi_changepoint',
    'config': make_changepoint([(1, 15, make_normal(80, 20)), (16, 35, make_normal(120, 25)), (36, 50, make_normal(100, 22))]),
    'description': '80->120->100'
}

# P09 Temp Spike/Dip
DISTRIBUTIONS[('p09_temp_spike_dip', 'v1_temp_surge')] = {
    'pattern_type': 'temp_spike',
    'config': make_changepoint([(1, 15, make_normal(100, 25)), (16, 25, make_normal(200, 35)), (26, 50, make_normal(100, 25))]),
    'description': '100->200->100'
}
DISTRIBUTIONS[('p09_temp_spike_dip', 'v2_temp_dip')] = {
    'pattern_type': 'temp_spike',
    'config': make_changepoint([(1, 15, make_normal(100, 25)), (16, 25, make_normal(50, 18)), (26, 50, make_normal(100, 25))]),
    'description': '100->50->100'
}
DISTRIBUTIONS[('p09_temp_spike_dip', 'v3_surge_new_normal')] = {
    'pattern_type': 'temp_spike',
    'config': make_changepoint([(1, 15, make_normal(100, 25)), (16, 25, make_normal(250, 40)), (26, 50, make_normal(120, 28))]),
    'description': '100->250->120'
}
DISTRIBUTIONS[('p09_temp_spike_dip', 'v4_dip_partial_recovery')] = {
    'pattern_type': 'temp_spike',
    'config': make_changepoint([(1, 15, make_normal(100, 25)), (16, 25, make_normal(40, 15)), (26, 50, make_normal(80, 22))]),
    'description': '100->40->80'
}

# P10 AR(1)
DISTRIBUTIONS[('p10_autocorrelated', 'v1_phi_0_7')] = {
    'pattern_type': 'ar1',
    'config': make_ar1(phi=0.7, c=30, sigma=20),
    'description': 'AR(1) phi=0.7'
}
DISTRIBUTIONS[('p10_autocorrelated', 'v2_phi_0_5')] = {
    'pattern_type': 'ar1',
    'config': make_ar1(phi=0.5, c=50, sigma=25),
    'description': 'AR(1) phi=0.5'
}
DISTRIBUTIONS[('p10_autocorrelated', 'v3_phi_0_3')] = {
    'pattern_type': 'ar1',
    'config': make_ar1(phi=0.3, c=70, sigma=30),
    'description': 'AR(1) phi=0.3'
}
DISTRIBUTIONS[('p10_autocorrelated', 'v4_phi_neg_0_3')] = {
    'pattern_type': 'ar1',
    'config': make_ar1(phi=-0.3, c=130, sigma=25),
    'description': 'AR(1) phi=-0.3'
}

def generate_single_demand(config, t, prev_demand=100):
    dist_type = config['type']
    if dist_type == 'normal':
        mu = config['mu'](t) if callable(config['mu']) else config['mu']
        sigma = config['sigma'](t) if callable(config['sigma']) else config['sigma']
        value = np.random.normal(mu, sigma)
    elif dist_type == 'uniform':
        value = np.random.uniform(config['a'], config['b'])
    elif dist_type == 'ar1':
        eps = np.random.normal(0, config['sigma'])
        value = config['phi'] * prev_demand + config['c'] + eps
    elif dist_type == 'changepoint':
        for (start, end, sub_config) in config['segments']:
            if start <= t <= end:
                return generate_single_demand(sub_config, t, prev_demand)
        _, _, sub_config = config['segments'][-1]
        return generate_single_demand(sub_config, t, prev_demand)
    else:
        raise ValueError(f"Unknown type: {dist_type}")
    return max(0, round(value))

def generate_trajectory(config, n_periods=50):
    demands = []
    prev = 100
    for t in range(1, n_periods + 1):
        d = generate_single_demand(config, t, prev)
        demands.append(d)
        prev = d
    return demands

def generate_train_samples(config, pattern_type, n_samples=5):
    if pattern_type == 'stationary':
        return [generate_single_demand(config, t=1) for _ in range(n_samples)]
    elif pattern_type in ['changepoint', 'multi_changepoint', 'temp_spike']:
        first_segment_config = config['segments'][0][2]
        return [generate_single_demand(first_segment_config, t=1) for _ in range(n_samples)]
    elif pattern_type in ['trend', 'seasonal', 'ar1']:
        samples = []
        prev = 100
        for t in range(1, n_samples + 1):
            d = generate_single_demand(config, t=t, prev_demand=prev)
            samples.append(d)
            prev = d
        return samples
    else:
        raise ValueError(f"Unknown pattern type: {pattern_type}")

def write_train_csv(path, samples):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write("exact_dates_chips(Regular),demand_chips(Regular)\n")
        for i, d in enumerate(samples, start=1):
            f.write(f"Period_{i},{d}\n")

def write_test_csv(path, demands, lead_time=0, profit=4, holding_cost=1):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write("exact_dates_chips(Regular),demand_chips(Regular),lead_time_chips(Regular),profit_chips(Regular),holding_cost_chips(Regular)\n")
        for i, d in enumerate(demands, start=1):
            f.write(f"Period_{i},{d},{lead_time},{profit},{holding_cost}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='synthetic_trajectory')
    parser.add_argument('--n-realizations', type=int, default=5)
    parser.add_argument('--n-periods', type=int, default=50)
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    output_dir = script_dir / args.output_dir
    
    print("=" * 70)
    print("SYNTHETIC BENCHMARK TRAJECTORY GENERATION")
    print("=" * 70)
    print(f"Output: {output_dir}")
    print(f"Distributions: {len(DISTRIBUTIONS)}, Realizations: {args.n_realizations}")
    print(f"Total: {len(DISTRIBUTIONS) * args.n_realizations} trajectories")
    print("=" * 70)
    
    total_folders = 0
    for (pattern, variant), dist_info in sorted(DISTRIBUTIONS.items()):
        config = dist_info['config']
        pattern_type = dist_info['pattern_type']
        
        train_seed = hash((args.seed, pattern, variant, "train")) % (2**32)
        np.random.seed(train_seed)
        train_samples = generate_train_samples(config, pattern_type)
        
        for r in range(1, args.n_realizations + 1):
            test_seed = hash((args.seed, pattern, variant, r)) % (2**32)
            np.random.seed(test_seed)
            test_demands = generate_trajectory(config, n_periods=args.n_periods)
            
            folder = output_dir / pattern / variant / f"r{r}"
            folder.mkdir(parents=True, exist_ok=True)
            write_train_csv(folder / "train.csv", train_samples)
            write_test_csv(folder / "test.csv", test_demands)
            total_folders += 1
        
        print(f"{pattern}/{variant}: train={train_samples}")
    
    print("\n" + "=" * 70)
    print(f"COMPLETE: {total_folders} folders created")
    actual = sum(1 for _ in output_dir.rglob("test.csv"))
    print(f"Verified: {actual} test.csv files")
    print("=" * 70)

if __name__ == "__main__":
    main()
