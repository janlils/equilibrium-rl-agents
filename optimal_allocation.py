from typing import List, Callable, Tuple

# ------------------------------------------------------------
# Rosenthal potential and diagnostics
# ------------------------------------------------------------

def potential(p_funcs: List[Callable[[int], float]], cost: List[float], alloc: List[int]) -> float:
    """
    Rosenthal potential:
        Phi(n) = sum_i sum_{k=1..n_i} (p_i(k) - c_i)
    """
    tot = 0.0
    for i, n in enumerate(alloc):
        s = 0.0
        for k in range(1, n + 1):
            s += p_funcs[i](k) - cost[i]
        tot += s
    return tot

def migration_pressure(p_funcs: List[Callable[[int], float]], cost: List[float], alloc: List[int]) -> float:
    """
    One-step best-response pressure:
        max_{i,j with n_j>0} [ π_i(n_i+1) - π_j(n_j) ],
    where π_i(k) = p_i(k) - c_i (marginal profit of the k-th entrant to market i).
    If <= 0, the allocation is one-step-stable (no single agent wants to move).
    """
    G = len(alloc)
    best = float("-inf")
    for i in range(G):
        pi_in = p_funcs[i](alloc[i] + 1) - cost[i]
        for j in range(G):
            if alloc[j] == 0:
                continue
            pj_out = p_funcs[j](alloc[j]) - cost[j]
            best = max(best, pi_in - pj_out)
    return best


# ------------------------------------------------------------
# Exact DP for free-mobility equilibrium (max potential Phi)
# ------------------------------------------------------------

def dp_potential_max(
    N: int,
    p_funcs: List[Callable[[int], float]],
    cost: List[float],
) -> Tuple[List[int], float]:
    """
    Exact DP for maximizing Rosenthal potential:
        Phi(n) = sum_i F_i(n_i),  where F_i(n) = sum_{k=1..n} (p_i(k) - c_i).
    dp[i][j] = best Phi using first i markets and exactly j people.
    Returns (allocation, Phi_max).
    """
    G = len(p_funcs)

    # Precompute cumulative F_i(n)
    F = [[0.0] * (N + 1) for _ in range(G)]
    for i, p in enumerate(p_funcs):
        cum = 0.0
        F[i][0] = 0.0
        for n in range(1, N + 1):
            cum += p(n) - cost[i]
            F[i][n] = cum

    NEG = -float("inf")
    dp = [[NEG] * (N + 1) for _ in range(G + 1)]
    choice = [[0] * (N + 1) for _ in range(G + 1)]
    dp[0][0] = 0.0

    for i in range(1, G + 1):
        for j in range(0, N + 1):
            best_val = NEG
            best_n = 0
            for n_add in range(0, j + 1):
                prev = dp[i - 1][j - n_add]
                if prev == NEG:
                    continue
                val = prev + F[i - 1][n_add]
                if val > best_val:
                    best_val = val
                    best_n = n_add
            dp[i][j] = best_val
            choice[i][j] = best_n

    # backtrack
    alloc = [0] * G
    j = N
    for i in range(G, 0, -1):
        alloc[i - 1] = choice[i][j]
        j -= alloc[i - 1]

    return alloc, dp[G][N]

def potential_normalized(
    N: int, p_funcs: List[Callable[[int], float]], cost: List[float], alloc: List[int]
) -> Tuple[float, float, float]:
    """
    Returns (Phi, Phi_max, Phi_ratio) where Phi_ratio = Phi / Phi_max in [0, 1]
    (defined as 1.0 when Phi_max == 0).
    """
    phi = potential(p_funcs, cost, alloc)
    _, phi_max = dp_potential_max(N, p_funcs, cost)
    ratio = (phi / phi_max) if phi_max != 0 else 1.0
    return phi, phi_max, ratio



if __name__ == "__main__":
    # --- problem setup ---
    N = 100
    cost: List[float] = [9.0, 8.0, 11.0, 10.0, 0.0]

    # Prices depend on total N and local n
    def p1(n: int) -> float: return (4 + N // 5) * 2 - n    
    def p2(n: int) -> float: return 10 + N // 5 - n           
    def p3(n: int) -> float: return N + 10 - (N // 5) * 4 - n 
    def p4(n: int) -> float: return 10 + N // 5 - n      
    def p5(n: int) -> float: return 1            

    # p_funcs: List[Callable[[int], float]] = [p1, p2, p3, p4]
    p_funcs: List[Callable[[int], float]] = [p1, p2, p3, p4, p5]

    # --- 1) Equilibrium (max potential) ---
    alloc_eq, phi_max = dp_potential_max(N, p_funcs, cost)
    print("Equilibrium (max potential) allocation:", alloc_eq)
    print("Phi_max:", phi_max)

    # --- 2) Quality of any given allocation (closeness to optimal potential) ---
    # alloc_test = [32, 23, 22, 23]
    alloc_test = [100, 0, 0, 0, 0]
    phi, phi_max_val, phi_ratio = potential_normalized(N, p_funcs, cost, alloc_test)
    press = migration_pressure(p_funcs, cost, alloc_test)

    print(f"\nTest allocation: {alloc_test}")
    print(f"  Phi(test) = {phi:.0f}, Phi_max = {phi_max_val:.0f}, Phi ratio = {phi_ratio:.4f}")
    print(f"  migration pressure (<=0 is stable): {press:.2f}")  
