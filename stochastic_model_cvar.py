# -*- coding: utf-8 -*-
"""
CVaR-Enhanced Two-Stage Stochastic Optimization Model
for Smart Home Cybersecurity Risk

This code extends the base stochastic model (S2) by incorporating
Conditional Value-at-Risk (CVaR) following Rockafellar & Uryasev (2000).

New variables added (only two):
  - eta_VaR  : scalar VaR threshold (free variable)
  - z_m      : excess loss above eta_VaR for each SAA sample m (>= 0)

The CVaR term replaces the pure expected-value objective with:
  min (1-lambda) * E[Loss] + lambda * CVaR_alpha

Setting lambda=0 reproduces the original S2 result exactly.
"""

import numpy as np
from scipy.stats import lognorm
import gurobipy as gp
from gurobipy import GRB

# ======================
# 1. SYSTEM PARAMETERS
# ======================

K   = [1, 2]          # Transaction types
L   = [1, 2, 3]       # Layers: edge=1, fog=2, cloud=3
N_l = {1: [1, 2, 3],  # Edge nodes
        2: [4, 5],     # Fog nodes
        3: [6]}        # Cloud node
N   = [n for nodes in N_l.values() for n in nodes]  # All nodes

# ======================
# 2. CONTROL OPTIONS
# ======================

S = {
    1:  "Network firewall",
    2:  "Application gateway",
    3:  "RBAC",
    4:  "HSRP",
    5:  "IDS/IPS",
    6:  "MFA",
    7:  "Network firewall+Application gateway",
    8:  "Network firewall+HSRP+RBAC",
    9:  "Network firewall+Application gateway+RBAC+HSRP",
    10: "Network firewall+Application gateway+RBAC+MFA+IDS/IPS",
    11: "Network firewall+Application gateway+RBAC+MFA+HSRP+IDS/IPS"
}

NS = {
    1: "Host firewall",
    2: "mTLS",
    3: "TPM",
    4: "Host firewall+mTLS",
    5: "Host firewall+TPM+mTLS"
}

# ======================
# 3. COST PARAMETERS ($)
# ======================

C_S = {
    1:  13754.08,   # Network firewall
    2:  11567.95,   # Application gateway
    3:  12990.00,   # RBAC
    4:  12795.60,   # HSRP
    5:  15690.00,   # IDS/IPS
    6:  15544.00,   # MFA
    7:  18322.03,   # Network firewall+Application gateway
    8:  21143.68,   # Network firewall+RBAC+HSRP
    9:  24711.63,   # Network firewall+Application gateway+RBAC+HSRP
    10:  7546.03,   # Network firewall+Application gateway+RBAC+MFA+IDS/IPS
    11: 35945.63    # Network firewall+Application gateway+RBAC+MFA+HSRP+IDS/IPS
}

C_NS = {
    1: 1248.72,     # Host firewall
    2: 1891.00,     # mTLS
    3: 1970.34,     # TPM
    4: 2139.72,     # Host firewall+mTLS
    5: 1237.06      # Host firewall+TPM+mTLS
}

# ======================
# 4. SYSTEM PARAMETERS
# ======================

D_SysNormal = {1: 30, 2: 30}   # Normal transaction rates (transactions/min)
C_avail     = {1: 100, 2: 100} # Availability breach cost per transaction
t_avail     = 20               # Time parameter (minutes)

rho_l  = {1: 90, 2: 70, 3: 55} # Layer processing capacities
pi_kl  = {(1,1):1, (1,2):1, (1,3):0,
           (2,1):1, (2,2):1, (2,3):1}

# Node capacities (transactions per minute)
Y_d_tx1 = {1: 30, 2: 30, 3: 30, 4: 35, 5: 35, 6:  0}
Y_d_tx2 = {1: 30, 2: 30, 3: 30, 4: 35, 5: 35, 6: 55}

# Breach loss amounts ($)
r_conf = {n: 518376.5957 for n in N}
r_int  = {n: 518376.5957 for n in N}
r_auth = {n: 451280.0000 for n in N}

# ======================
# 5. BREACH PROBABILITY PARAMETERS (piecewise linear)
# ======================

η_conf  = [0.555690/1000000, 0.517645/1000000]
ξ_conf  = [45.844057/100,    45.709050/100]
w_conf  = len(η_conf)

η_int   = [0.503994/1000000, 0.488700/1000000]
ξ_int   = [42.140788/100,    41.603242/100]
w_int   = len(η_int)

η_auth  = [0.496404/1000000, 0.505040/1000000]
ξ_auth  = [43.610619/100,    44.206732/100]
w_auth  = len(η_auth)

η_avail = [0.513798/1000000, 0.460665/1000000]
ξ_avail = [38.362751/100,    36.700180/100]
w_avail = len(η_avail)

# ======================
# 6. STOCHASTIC PARAMETERS
# ======================

mu_a, sigma_a = -0.009995, 0.205583  # ln(epsilon_avail) ~ N(mu_a, sigma_a^2)
E_conf = 0.999687   # E[epsilon_conf]
E_int  = 1.002789   # E[epsilon_int]
E_auth = 0.999370   # E[epsilon_auth]

# ======================
# 7. SAA SAMPLES
# ======================

M = 100              # Number of SAA scenarios
np.random.seed(42)   # Reproducibility
samples_avail = lognorm.rvs(s=sigma_a, scale=np.exp(mu_a), size=M)

# ======================
# 8. CVaR PARAMETERS  <-- NEW
# ======================
# alpha : confidence level  (0.95 = focus on worst 5% of scenarios)
# lambda_cvar : weight on CVaR term  (0 = pure expected value = original S2)
#
# Run for multiple (alpha, lambda) combinations to trace risk-return curve

alpha_list  = [0.95, 0.99]          # Confidence levels to test
lambda_list = [0.0, 0.25, 0.5, 0.75, 1.0]  # Risk-aversion weights

# ======================================================
# 9. CVaR-ENHANCED STOCHASTIC MODEL FUNCTION
# ======================================================

def solve_cvar_model(alpha, lambda_cvar, verbose=True):
    """
    Solve the CVaR-enhanced two-stage stochastic model S2.

    Parameters
    ----------
    alpha       : float  -- CVaR confidence level (e.g. 0.95)
    lambda_cvar : float  -- weight on CVaR vs expected loss (0 to 1)
    verbose     : bool   -- print detailed results

    Returns
    -------
    dict with objective value, CVaR, expected loss, security portfolio
    """

    try:
        model = gp.Model("S2_CVaR")
        model.setParam("OutputFlag", 0)  # Suppress Gurobi log; set to 1 to see solver output

        # ==========================================
        # STAGE 1: DECISION VARIABLES (unchanged)
        # ==========================================

        x_S  = model.addVars(S.keys(), vtype=GRB.BINARY, name="x_S")
        x_NS = model.addVars([(l,d,j) for l in L for d in N_l[l] for j in NS.keys()],
                              vtype=GRB.BINARY, name="x_NS")

        # Auxiliary variables for piecewise linear breach probabilities
        u_conf = model.addVars(N, lb=0, ub=100, name="u_conf")
        u_int  = model.addVars(N, lb=0, ub=100, name="u_int")
        u_auth = model.addVars(N, lb=0, ub=100, name="u_auth")

        # Total security cost per node
        C_total = model.addVars(N, lb=0, name="C_total")

        # ==========================================
        # STAGE 2: SCENARIO VARIABLES (unchanged)
        # ==========================================

        D_breach         = model.addVars(K, range(M), lb=0,
                                         ub=max(D_SysNormal.values()), name="D_breach")
        capacity_red_tx1 = model.addVars(range(M), name="cap_red_tx1")
        capacity_red_tx2 = model.addVars(range(M), name="cap_red_tx2")
        total_processed  = model.addVars(K, range(M), name="total_processed")
        effective_cap    = model.addVars(range(M), name="effective_cap")

        # ==========================================
        # CVaR VARIABLES  <-- NEW (only 2 additions)
        # ==========================================

        # eta_VaR: the VaR threshold (Value-at-Risk at level alpha)
        # It is a FREE variable (can be positive or negative)
        eta_VaR = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="eta_VaR")

        # z_m: excess of scenario loss above eta_VaR, one per scenario
        # Must be >= 0 (enforced via lb and constraint below)
        z = model.addVars(range(M), lb=0, name="z")

        # ==========================================
        # OBJECTIVE FUNCTION  <-- MODIFIED
        # ==========================================

        # --- Component 1: Stage 1 control costs (unchanged) ---
        control_costs = (
            gp.quicksum(C_S[i]  * x_S[i]       for i in S.keys()) +
            gp.quicksum(C_NS[j] * x_NS[l, d, j]
                        for l in L for d in N_l[l] for j in NS.keys())
        )

        # --- Component 2: Expected breach costs conf/int/auth (unchanged) ---
        breach_costs = (
            gp.quicksum(r_conf[d] * E_conf * u_conf[d] for d in N) +
            gp.quicksum(r_int[d]  * E_int  * u_int[d]  for d in N) +
            gp.quicksum(r_auth[d] * E_auth * u_auth[d] for d in N)
        )

        # --- Component 3: SAA availability loss per scenario ---
        # This is the stage-2 loss L_m expressed directly using D_breach[k,m]
        # (L_m is NOT a variable -- it is this expression substituted inline)
        avail_loss_per_scenario = {
            m: gp.quicksum(
                C_avail[k] * (D_SysNormal[k] - D_breach[k, m]) * t_avail
                for k in K
            )
            for m in range(M)
        }

        # Expected availability loss (SAA approximation, as in original S2)
        expected_avail_loss = (1/M) * gp.quicksum(
            avail_loss_per_scenario[m] for m in range(M)
        )

        # --- Component 4: CVaR term  <-- NEW ---
        # CVaR_alpha = eta_VaR + (1/(1-alpha)) * (1/M) * sum_m z_m
        cvar_term = eta_VaR + (1.0 / (1.0 - alpha)) * (1.0/M) * gp.quicksum(
            z[m] for m in range(M)
        )

        # --- Combined objective ---
        # lambda_cvar=0 --> pure expected value (original S2)
        # lambda_cvar=1 --> pure CVaR (fully risk-averse)
        model.setObjective(
            control_costs + breach_costs +
            (1 - lambda_cvar) * expected_avail_loss +
            lambda_cvar       * cvar_term,
            GRB.MINIMIZE
        )

        # ==========================================
        # CONSTRAINTS (original S2 -- unchanged)
        # ==========================================

        # Total security cost per node
        for d in N:
            model.addConstr(
                C_total[d] == (
                    gp.quicksum(C_S[i] * x_S[i] for i in S.keys()) +
                    gp.quicksum(C_NS[j] * x_NS[l, d, j]
                                for l in L for j in NS.keys() if d in N_l[l])
                ),
                name=f"total_cost_{d}"
            )

        # Piecewise linear breach probability constraints
        for d in N:
            for m in range(w_conf):
                model.addConstr(u_conf[d] >= η_conf[m] * C_total[d] + ξ_conf[m],
                                name=f"conf_{d}_{m}")
            for m in range(w_int):
                model.addConstr(u_int[d]  >= η_int[m]  * C_total[d] + ξ_int[m],
                                name=f"int_{d}_{m}")
            for m in range(w_auth):
                model.addConstr(u_auth[d] >= η_auth[m] * C_total[d] + ξ_auth[m],
                                name=f"auth_{d}_{m}")

        # Scenario-specific capacity constraints
        for m in range(M):
            model.addConstr(
                capacity_red_tx1[m] == gp.quicksum(
                    Y_d_tx1[d] * (η_avail[0] * C_total[d] + ξ_avail[0]) * samples_avail[m]
                    for d in N
                ), name=f"cap_red_tx1_{m}"
            )
            model.addConstr(
                capacity_red_tx2[m] == gp.quicksum(
                    Y_d_tx2[d] * (η_avail[0] * C_total[d] + ξ_avail[0]) * samples_avail[m]
                    for d in N
                ), name=f"cap_red_tx2_{m}"
            )

            eff_tx1 = sum(rho_l.values()) - capacity_red_tx1[m]
            eff_tx2 = sum(rho_l.values()) - capacity_red_tx2[m]

            for k in K:
                if k == 1:
                    model.addConstr(
                        total_processed[k, m] == gp.quicksum(
                            D_breach[k, m] * pi_kl.get((k, l), 0) for l in L),
                        name=f"total_proc_tx1_{m}"
                    )
                    model.addConstr(total_processed[k, m] <= eff_tx1,
                                   name=f"cap_tx1_{m}")
                    model.addConstr(
                        D_breach[k, m] <= D_SysNormal[k] * (eff_tx1 / sum(rho_l.values())),
                        name=f"cap_adj_tx1_{m}"
                    )
                else:
                    model.addConstr(
                        total_processed[k, m] == gp.quicksum(
                            D_breach[k, m] * pi_kl.get((k, l), 0) for l in L),
                        name=f"total_proc_tx2_{m}"
                    )
                    model.addConstr(total_processed[k, m] <= eff_tx2,
                                   name=f"cap_tx2_{m}")
                    model.addConstr(
                        D_breach[k, m] <= D_SysNormal[k] * (eff_tx2 / sum(rho_l.values())),
                        name=f"cap_adj_tx2_{m}"
                    )

            model.addConstr(
                effective_cap[m] == sum(rho_l.values()) - capacity_red_tx1[m],
                name=f"effective_cap_{m}"
            )

        # Exactly one system-level control
        model.addConstr(gp.quicksum(x_S[i] for i in S.keys()) == 1,
                        name="system_control_selection")

        # Exactly one node-level control per node
        for l in L:
            for d in N_l[l]:
                model.addConstr(
                    gp.quicksum(x_NS[l, d, j] for j in NS.keys()) == 1,
                    name=f"node_control_{d}"
                )

        # ==========================================
        # CVaR CONSTRAINTS  <-- NEW
        # z_m >= L_m - eta_VaR  for all m
        # L_m is the stage-2 availability loss expression (substituted directly)
        # ==========================================

        for m in range(M):
            # L_m = availability loss in scenario m
            # = sum_k [ C_avail[k] * (D_SysNormal[k] - D_breach[k,m]) * t_avail ]
            # (Note: breach costs conf/int/auth are scenario-independent,
            #  so only availability loss varies by scenario here)
            L_m = avail_loss_per_scenario[m]  # reuse the expression defined above

            model.addConstr(
                z[m] >= L_m - eta_VaR,
                name=f"cvar_z_{m}"
            )
            # z[m] >= 0 is already enforced by lb=0 in addVar

        # ==========================================
        # SOLVE
        # ==========================================

        model.optimize()

        # ==========================================
        # RESULTS
        # ==========================================

        if model.status == GRB.OPTIMAL:

            # Extract scenario losses
            scenario_losses = [
                sum(C_avail[k] * (D_SysNormal[k] - D_breach[k, m].X) * t_avail
                    for k in K)
                for m in range(M)
            ]

            # Compute CVaR from results
            eta_val   = eta_VaR.X
            cvar_val  = eta_val + (1/(1-alpha)) * np.mean([z[m].X for m in range(M)])
            exp_loss  = np.mean(scenario_losses)

            # Transaction rates
            tx1_rates = [D_breach[1, m].X for m in range(M)]
            tx2_rates = [D_breach[2, m].X for m in range(M)]

            # Security portfolio
            sys_selected  = [S[i]  for i in S  if x_S[i].X  > 0.5]
            node_selected = [NS[j] for l in L for d in N_l[l]
                             for j in NS if x_NS[l, d, j].X > 0.5]

            total_control_cost = (
                sum(C_S[i]  * x_S[i].X  for i in S) +
                sum(C_NS[j] * x_NS[l, d, j].X
                    for l in L for d in N_l[l] for j in NS)
            )

            if verbose:
                print(f"\n{'='*60}")
                print(f"CVaR MODEL RESULTS  |  alpha={alpha}  |  lambda={lambda_cvar}")
                print(f"{'='*60}")

                print("\n-- Security Portfolio --")
                print(f"  System-level  : {sys_selected}")
                print(f"  Node-level    : {node_selected[0] if node_selected else 'None'}")

                print("\n-- Cost Breakdown --")
                print(f"  Control Cost  : ${total_control_cost:,.2f}")
                print(f"  Expected Loss : ${exp_loss:,.2f}")
                print(f"  VaR (eta)     : ${eta_val:,.2f}")
                print(f"  CVaR_{int(alpha*100)}%    : ${cvar_val:,.2f}")
                print(f"  Obj Value     : ${model.ObjVal:,.2f}")

                print("\n-- Transaction Rate Statistics --")
                print(f"  {'Metric':<10} {'Tx1':>10} {'Tx2':>10}")
                print(f"  {'Mean':<10} {np.mean(tx1_rates):>10.2f} {np.mean(tx2_rates):>10.2f}")
                print(f"  {'Std Dev':<10} {np.std(tx1_rates):>10.2f}  {np.std(tx2_rates):>10.2f}")
                print(f"  {'Min':<10} {min(tx1_rates):>10.2f}  {min(tx2_rates):>10.2f}")
                print(f"  {'Max':<10} {max(tx1_rates):>10.2f}  {max(tx2_rates):>10.2f}")

            return {
                "alpha":              alpha,
                "lambda":             lambda_cvar,
                "obj_value":          model.ObjVal,
                "expected_loss":      exp_loss,
                "cvar":               cvar_val,
                "eta_VaR":            eta_val,
                "control_cost":       total_control_cost,
                "sys_portfolio":      sys_selected,
                "node_portfolio":     node_selected,
                "tx1_mean":           np.mean(tx1_rates),
                "tx2_mean":           np.mean(tx2_rates),
                "tx1_min":            min(tx1_rates),
                "tx2_min":            min(tx2_rates),
            }

        else:
            print(f"  Optimization failed with status: {model.status}")
            return None

    except gp.GurobiError as e:
        print(f"Gurobi error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


# ======================================================
# 10. RUN EXPERIMENTS ACROSS (alpha, lambda) GRID
# ======================================================

print("Running CVaR experiments across (alpha, lambda) combinations...")
print("Note: lambda=0.0 reproduces original S2 result exactly\n")

results = []

for alpha in alpha_list:
    for lam in lambda_list:
        res = solve_cvar_model(alpha=alpha, lambda_cvar=lam, verbose=True)
        if res:
            results.append(res)

# ======================================================
# 11. SUMMARY TABLE
# ======================================================

print("\n\n" + "="*90)
print("SUMMARY TABLE: CVaR Results Across Risk-Aversion Levels")
print("="*90)
print(f"{'alpha':>6} {'lambda':>8} {'Exp.Loss ($)':>14} {'CVaR ($)':>14} "
      f"{'Control Cost ($)':>17} {'Obj Value ($)':>14}")
print("-"*90)

for r in results:
    print(f"{r['alpha']:>6.2f} {r['lambda']:>8.2f} "
          f"{r['expected_loss']:>14,.2f} "
          f"{r['cvar']:>14,.2f} "
          f"{r['control_cost']:>17,.2f} "
          f"{r['obj_value']:>14,.2f}")

print("="*90)

# ======================================================
# 12. RISK-RETURN TRADEOFF PLOT
# ======================================================

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for alpha_val in alpha_list:
    subset = [r for r in results if r["alpha"] == alpha_val]
    lambdas    = [r["lambda"]        for r in subset]
    exp_losses = [r["expected_loss"] for r in subset]
    cvar_vals  = [r["cvar"]          for r in subset]

    axes[0].plot(lambdas, exp_losses, marker='o', label=f"alpha={alpha_val}")
    axes[1].plot(lambdas, cvar_vals,  marker='s', label=f"alpha={alpha_val}")

axes[0].set_title("Expected Availability Loss vs Lambda")
axes[0].set_xlabel("Lambda (Risk-Aversion Weight)")
axes[0].set_ylabel("Expected Loss ($)")
axes[0].legend()
axes[0].grid(True, alpha=0.4)

axes[1].set_title("CVaR vs Lambda")
axes[1].set_xlabel("Lambda (Risk-Aversion Weight)")
axes[1].set_ylabel("CVaR ($)")
axes[1].legend()
axes[1].grid(True, alpha=0.4)

plt.suptitle("CVaR Risk-Return Tradeoff: Effect of Risk Aversion on Tail Risk",
             fontsize=13)
plt.tight_layout()
plt.savefig("cvar_risk_return_tradeoff.png", dpi=300, bbox_inches='tight')
plt.show()
print("\nPlot saved to: cvar_risk_return_tradeoff.png")
