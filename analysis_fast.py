import numpy as np
import sys
sys.path.insert(0, '.')

from solver.optimal_timing import ShiftTimingSolver, Decision

RISK_AVERSION = 0.05
N_PATHS = 2000  # Fewer paths for speed

print(f"Risk aversion = {RISK_AVERSION}, n_paths = {N_PATHS}")
print("="*60)

results = []

# 1. Moneyness (fewer points)
print("\n1. MONEYNESS (30d expiry, 14d deadline, 6% vol)")
for spot in [1.06, 1.10, 1.14]:
    moneyness = (spot - 1.10) / 1.10 * 100
    solver = ShiftTimingSolver(
        spot=spot, strike=1.10, vol=0.06,
        expiry_days=30, shift_bps=5,
        risk_aversion=RISK_AVERSION,
        deadline_days=14, n_paths=N_PATHS, seed=42,
    )
    r = solver.solve()
    ratio = r.expected_cost_wait / r.cost_now
    print(f"  Spot={spot:.2f} ({moneyness:+.1f}%): E[Cost]/Cost_now={ratio:.3f}, Decision={r.decision.value}")
    results.append(('moneyness', spot, r.decision.value))

# 2. Volatility (fewer points)
print("\n2. VOLATILITY (ATM, 30d expiry, 14d deadline)")
for vol in [0.04, 0.08, 0.12]:
    solver = ShiftTimingSolver(
        spot=1.10, strike=1.10, vol=vol,
        expiry_days=30, shift_bps=5,
        risk_aversion=RISK_AVERSION,
        deadline_days=14, n_paths=N_PATHS, seed=42,
    )
    r = solver.solve()
    ratio = r.expected_cost_wait / r.cost_now
    print(f"  Vol={vol*100:.0f}%: E[Cost]/Cost_now={ratio:.3f}, Decision={r.decision.value}")
    results.append(('vol', vol, r.decision.value))

# 3. Deadline (how long to wait)
print("\n3. DEADLINE (ATM, 30d expiry, 6% vol)")
for deadline in [7, 14, 21]:
    days_to_wait = 30 - deadline
    solver = ShiftTimingSolver(
        spot=1.10, strike=1.10, vol=0.06,
        expiry_days=30, shift_bps=5,
        risk_aversion=RISK_AVERSION,
        deadline_days=deadline, n_paths=N_PATHS, seed=42,
    )
    r = solver.solve()
    ratio = r.expected_cost_wait / r.cost_now
    print(f"  Deadline={deadline}d (wait {days_to_wait}d): E[Cost]/Cost_now={ratio:.3f}, Decision={r.decision.value}")
    results.append(('deadline', deadline, r.decision.value))

# 4. Expiry (with fixed 14d deadline)
print("\n4. EXPIRY (ATM, 6% vol, 14d deadline)")
for expiry in [21, 45, 90]:
    days_to_wait = expiry - 14
    solver = ShiftTimingSolver(
        spot=1.10, strike=1.10, vol=0.06,
        expiry_days=expiry, shift_bps=5,
        risk_aversion=RISK_AVERSION,
        deadline_days=14, n_paths=N_PATHS, seed=42,
    )
    r = solver.solve()
    ratio = r.expected_cost_wait / r.cost_now
    print(f"  Expiry={expiry}d (wait {days_to_wait}d): E[Cost]/Cost_now={ratio:.3f}, Decision={r.decision.value}")
    results.append(('expiry', expiry, r.decision.value))

print("\n" + "="*60)
wait_count = sum(1 for r in results if r[2] == 'WAIT')
print(f"WAIT recommendations: {wait_count}/{len(results)}")
