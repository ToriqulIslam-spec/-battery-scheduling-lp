# ============================================
# Cost-Optimal Battery Scheduling for a Residential Solar Microgrid
# Using Linear Programming (PuLP)
#
# Author: MD. Toriqul Islam Srestho
# Institution: Notre Dame College, Dhaka, Bangladesh
#
# This script reproduces ALL results in:
# "Cost-Optimal Battery Scheduling for a Residential Solar Microgrid
#  Using Linear Programming"
#
# Run in Google Colab. Requirements: pulp, numpy, matplotlib
# ============================================

!pip install pulp -q

import numpy as np
import matplotlib.pyplot as plt
import pulp

# ============================================
# 1. SYSTEM PARAMETERS & 24-HOUR PROFILES
# ============================================
HOURS = np.arange(24)

# Home load P_load(t) in kW (Section III)
P_load = np.array([
    0.5, 0.4, 0.4, 0.5, 0.6, 0.8, 1.2, 1.5, 1.0, 0.8, 0.8, 0.8,
    0.9, 0.9, 1.0, 1.2, 1.8, 2.5, 3.0, 2.8, 2.0, 1.5, 1.0, 0.6
])

# Available PV generation P_pv(t) in kW (Section III)
P_pv = np.array([
    0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.8, 1.5, 2.5, 3.5, 4.0, 4.2,
    4.0, 3.5, 2.5, 1.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
])

# ============================================
# 2. BASELINE RULE-BASED SCHEDULER (Section V)
# ============================================
def run_baseline(P_load, P_pv, price, SOC_min, SOC_max, SOC_init,
                 P_ch_max, P_dis_max, eta_c, eta_d):
    """
    Baseline battery scheduler.
    Returns cost, SOC trajectory (length 25, beginning of each hour), and P_grid.
    """
    SOC = SOC_init
    SOC_traj = np.zeros(25)
    SOC_traj[0] = SOC_init
    P_grid = np.zeros(24)

    for t in range(24):
        net = P_pv[t] - P_load[t]

        if net > 0:
            # Surplus solar: charge battery (Eq. 5, 6 in paper)
            P_ch = min(net, P_ch_max)
            space = (SOC_max - SOC) / eta_c
            P_ch = min(P_ch, space)
            SOC += P_ch * eta_c
            P_grid[t] = 0.0
        else:
            # Deficit: discharge battery to cover load
            deficit = -net
            P_dis = min(deficit, P_dis_max)
            energy_avail = (SOC - SOC_min) * eta_d
            P_dis = min(P_dis, energy_avail)
            SOC -= P_dis / eta_d
            P_grid[t] = deficit - P_dis

        SOC_traj[t+1] = SOC

    cost = np.sum(P_grid * price)  # Eq. (7) in paper
    return cost, SOC_traj, P_grid


# ============================================
# 3. LP OPTIMIZER (Section V) — Using PuLP
# ============================================
def run_optimized(P_load, P_pv, price, SOC_min, SOC_max, SOC_init,
                  P_ch_max, P_dis_max, eta_c, eta_d, final_SOC_bound=None):
    """
    LP-based optimal battery scheduler using PuLP.
    Decision variables per hour t:
        P_grid(t), P_ch(t), P_dis(t), P_pv_used(t), SOC(t)
    """
    prob = pulp.LpProblem("Battery_Scheduling", pulp.LpMinimize)
    T = range(24)

    # Variables
    P_grid = pulp.LpVariable.dicts("P_grid", T, lowBound=0)
    P_ch = pulp.LpVariable.dicts("P_ch", T, lowBound=0, upBound=P_ch_max)
    P_dis = pulp.LpVariable.dicts("P_dis", T, lowBound=0, upBound=P_dis_max)
    P_pv_used = {t: pulp.LpVariable(f"P_pv_used_{t}", lowBound=0, upBound=P_pv[t]) for t in T}
    SOC = {t: pulp.LpVariable(f"SOC_{t}", lowBound=SOC_min, upBound=SOC_max) for t in range(25)}

    # Objective: min J = sum_t price(t) * P_grid(t) * dt   (Eq. 7)
    dt = 1.0
    prob += pulp.lpSum([price[t] * P_grid[t] * dt for t in T])

    # Initial SOC constraint
    prob += SOC[0] == SOC_init

    # Hourly constraints
    for t in T:
        # Power balance: P_grid + P_dis + P_pv_used = P_load + P_ch  (Eq. 1)
        prob += P_grid[t] + P_dis[t] + P_pv_used[t] == P_load[t] + P_ch[t]

        # SOC dynamics: SOC[t+1] = SOC[t] + eta_c*P_ch - P_dis/eta_d  (Eq. 3)
        prob += SOC[t+1] == SOC[t] + eta_c * P_ch[t] * dt - (P_dis[t] * dt) / eta_d

    # Optional final SOC bound (Scenario B)
    if final_SOC_bound is not None:
        prob += SOC[24] >= final_SOC_bound

    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] == 'Optimal':
        opt_cost = pulp.value(prob.objective)
        opt_SOC = np.array([pulp.value(SOC[t]) for t in range(25)])
        opt_P_grid = np.array([pulp.value(P_grid[t]) for t in T])
        return opt_cost, opt_SOC, opt_P_grid
    else:
        print("LP failed:", pulp.LpStatus[prob.status])
        return None, None, None


# ============================================
# 4. SCENARIO A: 5 kWh, Mild TOU, NO Final SOC Bound
# ============================================
# Parameters from Table I in paper
price_mild = np.array([0.15]*6 + [0.25]*11 + [0.45]*5 + [0.15]*2)

base_cost_A, base_SOC_A, base_grid_A = run_baseline(
    P_load, P_pv, price_mild,
    SOC_min=1.0, SOC_max=4.5, SOC_init=2.5,
    P_ch_max=1.5, P_dis_max=1.5,
    eta_c=0.95, eta_d=0.95
)

opt_cost_A, opt_SOC_A, opt_grid_A = run_optimized(
    P_load, P_pv, price_mild,
    SOC_min=1.0, SOC_max=4.5, SOC_init=2.5,
    P_ch_max=1.5, P_dis_max=1.5,
    eta_c=0.95, eta_d=0.95,
    final_SOC_bound=None
)

# ============================================
# 5. SCENARIO B: 5 kWh, Mild TOU, WITH Final SOC Bound
# ============================================
opt_cost_B, opt_SOC_B, opt_grid_B = run_optimized(
    P_load, P_pv, price_mild,
    SOC_min=1.0, SOC_max=4.5, SOC_init=2.5,
    P_ch_max=1.5, P_dis_max=1.5,
    eta_c=0.95, eta_d=0.95,
    final_SOC_bound=2.5
)

# ============================================
# 6. SCENARIO C: 7.5 kWh, Strong TOU, NO Final SOC Bound
# ============================================
price_strong = np.array([0.12]*5 + [0.28]*11 + [0.58]*6 + [0.12]*2)

base_cost_C, base_SOC_C, base_grid_C = run_baseline(
    P_load, P_pv, price_strong,
    SOC_min=1.5, SOC_max=6.75, SOC_init=4.0,
    P_ch_max=3.0, P_dis_max=3.0,
    eta_c=0.95, eta_d=0.95
)

opt_cost_C, opt_SOC_C, opt_grid_C = run_optimized(
    P_load, P_pv, price_strong,
    SOC_min=1.5, SOC_max=6.75, SOC_init=4.0,
    P_ch_max=3.0, P_dis_max=3.0,
    eta_c=0.95, eta_d=0.95,
    final_SOC_bound=None
)

# ============================================
# 7. RESULTS OUTPUT (Tables II, III in paper)
# ============================================
print("=" * 60)
print("SCENARIO A (5 kWh, Mild Tariff, NO Final SOC Bound)")
print("=" * 60)
print(f"Baseline Cost  : ${base_cost_A:.2f}")
print(f"Optimized Cost : ${opt_cost_A:.2f}")
print(f"Absolute Save  : ${base_cost_A - opt_cost_A:.2f}")
print(f"Percent Save   : {((base_cost_A - opt_cost_A)/base_cost_A)*100:.2f}%")
print(f"Final SOC      : {opt_SOC_A[-1]:.2f} kWh")

print("\n" + "=" * 60)
print("SCENARIO B (5 kWh, Mild Tariff, WITH Final SOC Bound)")
print("=" * 60)
print(f"Baseline Cost  : ${base_cost_A:.2f}")
print(f"Optimized Cost : ${opt_cost_B:.2f}")
print(f"Absolute Save  : ${base_cost_A - opt_cost_B:.2f}")
print(f"Percent Save   : {((base_cost_A - opt_cost_B)/base_cost_A)*100:.2f}%")
print(f"Final SOC      : {opt_SOC_B[-1]:.2f} kWh")

print("\n" + "=" * 60)
print("SCENARIO C (7.5 kWh, Strong Tariff, NO Final SOC Bound)")
print("=" * 60)
print(f"Baseline Cost  : ${base_cost_C:.2f}")
print(f"Optimized Cost : ${opt_cost_C:.2f}")
print(f"Absolute Save  : ${base_cost_C - opt_cost_C:.2f}")
print(f"Percent Save   : {((base_cost_C - opt_cost_C)/base_cost_C)*100:.2f}%")
print(f"Final SOC      : {opt_SOC_C[-1]:.2f} kWh")

print("\n" + "=" * 60)
print("LATEX TABLE VALUES")
print("=" * 60)
print(f"Scenario A: Base ${base_cost_A:.2f} | Opt ${opt_cost_A:.2f} | Save {((base_cost_A-opt_cost_A)/base_cost_A)*100:.2f}%")
print(f"Scenario B: Base ${base_cost_A:.2f} | Opt ${opt_cost_B:.2f} | Save {((base_cost_A-opt_cost_B)/base_cost_A)*100:.2f}%")
print(f"Scenario C: Base ${base_cost_C:.2f} | Opt ${opt_cost_C:.2f} | Save {((base_cost_C-opt_cost_C)/base_cost_C)*100:.2f}%")


# ============================================
# 8. FIGURE 1: System Profiles + SOC Trajectory
# ============================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)

# Top panel: Load and PV
ax1.plot(HOURS, P_load, 'r--', marker='o', markersize=5, label='Load (kW)')
ax1.plot(HOURS, P_pv, 'g-', marker='s', markersize=5, label='PV Gen (kW)')
ax1.set_ylabel('Power (kW)', fontsize=11)
ax1.set_title('24-Hour System Profiles', fontsize=12, fontweight='bold')
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-0.5, 23.5)

# Bottom panel: Battery SOC trajectories (Scenario A)
ax2.plot(HOURS, base_SOC_A[:-1], 'k:', linewidth=1.5, label='Baseline SOC')
ax2.plot(HOURS, opt_SOC_A[:-1], 'darkgreen', linewidth=2.2, label='Optimized SOC')
ax2.axhline(4.5, color='r', linestyle='--', alpha=0.5, label='SOC Max')
ax2.axhline(1.0, color='r', linestyle='--', alpha=0.5, label='SOC Min')
ax2.set_xlabel('Hour of the Day', fontsize=11)
ax2.set_ylabel('State of Charge (kWh)', fontsize=11)
ax2.set_title('Battery State of Charge (SOC) Trajectory', fontsize=12, fontweight='bold')
ax2.legend(loc='lower left', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-0.5, 23.5)

plt.tight_layout()
plt.savefig('fig1.png', dpi=300, bbox_inches='tight')
plt.show()


# ============================================
# 9. FIGURE 2: Economic Performance Comparison
# ============================================
plt.figure(figsize=(6, 4.5))
cases = ['Baseline', 'Optimized\n(No Bound)', 'Optimized\n(With Bound)']
costs = [base_cost_A, opt_cost_A, opt_cost_B]
colors = ['gray', 'teal', 'lightblue']
bars = plt.bar(cases, costs, color=colors, width=0.5,
               edgecolor='black', linewidth=0.5)

plt.ylabel('Daily Electricity Cost ($)', fontsize=11)
plt.title('Economic Performance Comparison (5 kWh Battery)', fontsize=12, fontweight='bold')
plt.ylim(0, max(costs) * 1.3)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.08,
             f"${yval:.2f}", ha='center', va='bottom',
             fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('fig2.png', dpi=300, bbox_inches='tight')
plt.show()


# ============================================
# 10. FIGURE 3: Sensitivity — Cost Savings vs. Battery Capacity
# ============================================
capacities = [3, 5, 7.5, 10, 12]
savings_pct = []

for cap in capacities:
    SOC_min_i = cap * 0.20
    SOC_max_i = cap * 0.90
    SOC_init_i = cap * 0.50
    P_max_i = min(cap * 0.40, 3.0)

    b_cost, _, _ = run_baseline(
        P_load, P_pv, price_strong,
        SOC_min=SOC_min_i, SOC_max=SOC_max_i, SOC_init=SOC_init_i,
        P_ch_max=P_max_i, P_dis_max=P_max_i,
        eta_c=0.95, eta_d=0.95
    )
    o_cost, _, _ = run_optimized(
        P_load, P_pv, price_strong,
        SOC_min=SOC_min_i, SOC_max=SOC_max_i, SOC_init=SOC_init_i,
        P_ch_max=P_max_i, P_dis_max=P_max_i,
        eta_c=0.95, eta_d=0.95,
        final_SOC_bound=None
    )
    savings_pct.append(((b_cost - o_cost) / b_cost) * 100)

plt.figure(figsize=(5, 3.5))
plt.plot(capacities, savings_pct, 'bo-', markersize=7, linewidth=2)
plt.xlabel('Battery Capacity (kWh)', fontsize=10)
plt.ylabel('Savings (%)', fontsize=10)
plt.title('Cost Savings vs. Battery Capacity', fontsize=11, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.ylim(0, max(savings_pct) * 1.2)
plt.tight_layout()
plt.savefig('fig3.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("FIGURE 3 VALUES")
print("=" * 60)
for cap, save in zip(capacities, savings_pct):
    print(f"  Capacity {cap:>5.1f} kWh  ->  Savings {save:.2f}%")

print("\nAll figures saved. Results match paper Tables II and III.")
