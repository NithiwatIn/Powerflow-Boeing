# microgrid_project/simulation/newtonrapson_loadflow.py

import numpy as np
import pandas as pd

def run_newton_raphson(bus_data: pd.DataFrame, gen_data: pd.DataFrame, load_data: pd.DataFrame, 
                       y_bus: np.ndarray, base_mva: float = 100.0, 
                       max_iter: int = 20, tolerance: float = 1e-5, 
                       perform_pf_dispatch: bool = True) -> tuple: # perform_pf_dispatch is no longer used but kept for compatibility
    
    G = y_bus.real
    B = y_bus.imag
    num_buses = len(bus_data)
    V = bus_data['V_init'].values.copy()
    delta = np.deg2rad(bus_data['Angle_init'].values.copy())

    bus_types = bus_data['Type'].values
    slack_bus_indices = np.where(bus_types == 1)[0]
    if len(slack_bus_indices) == 0: raise ValueError("No Slack Bus (Type 1) found.")
    
    pv_bus_indices = np.where(bus_types == 2)[0]
    pq_bus_indices = np.where(bus_types == 3)[0]

    non_slack_indices = np.sort(np.concatenate([pv_bus_indices, pq_bus_indices]))
    pq_indices = np.sort(pq_bus_indices)

    # This function is now a PURE SOLVER. It uses the Pg values as provided.
    bus_ids = bus_data['BusID'].values
    pg_per_bus = gen_data.groupby('BusID')['Pg_MW'].sum().reindex(bus_ids, fill_value=0)
    qg_per_bus = gen_data.groupby('BusID')['Qg_MVAR'].sum().reindex(bus_ids, fill_value=0)
    pd_per_bus = load_data.groupby('BusID')['Pd_MW'].sum().reindex(bus_ids, fill_value=0)
    qd_per_bus = load_data.groupby('BusID')['Qd_MVAR'].sum().reindex(bus_ids, fill_value=0)

    P_sch = (pg_per_bus - pd_per_bus).values / base_mva
    Q_sch = (qg_per_bus - qd_per_bus).values / base_mva

    is_converged = False
    iteration = 0
    for iteration in range(max_iter):
        V_complex = V * np.exp(1j * delta)
        S_calc_complex = V_complex * np.conj(y_bus @ V_complex)
        P_calc = S_calc_complex.real
        Q_calc = S_calc_complex.imag

        mismatch_P = (P_sch - P_calc)[non_slack_indices]
        mismatch_Q = (Q_sch - Q_calc)[pq_indices]
        mismatch_vector = np.concatenate([mismatch_P, mismatch_Q])
        
        if np.max(np.abs(mismatch_vector)) < tolerance:
            is_converged = True; break

        # Jacobian Calculation (Full code)
        J11 = np.zeros((len(non_slack_indices), len(non_slack_indices)))
        J12 = np.zeros((len(non_slack_indices), len(pq_indices)))
        J21 = np.zeros((len(pq_indices), len(non_slack_indices)))
        J22 = np.zeros((len(pq_indices), len(pq_indices)))
        for i_idx, i in enumerate(non_slack_indices):
            for k_idx, k in enumerate(non_slack_indices):
                if i == k: J11[i_idx, k_idx] = -Q_calc[i] - V[i]**2 * B[i, i]
                else: angle_ik = delta[i] - delta[k]; J11[i_idx, k_idx] = V[i] * V[k] * (G[i, k] * np.sin(angle_ik) - B[i, k] * np.cos(angle_ik))
        for i_idx, i in enumerate(non_slack_indices):
            for k_idx, k in enumerate(pq_indices):
                if i == k: J12[i_idx, k_idx] = P_calc[i] / V[i] + V[i] * G[i, i]
                else: angle_ik = delta[i] - delta[k]; J12[i_idx, k_idx] = V[i] * (G[i, k] * np.cos(angle_ik) + B[i, k] * np.sin(angle_ik))
        for i_idx, i in enumerate(pq_indices):
            for k_idx, k in enumerate(non_slack_indices):
                if i == k: J21[i_idx, k_idx] = P_calc[i] - V[i]**2 * G[i, i]
                else: angle_ik = delta[i] - delta[k]; J21[i_idx, k_idx] = -V[i] * V[k] * (G[i, k] * np.cos(angle_ik) + B[i, k] * np.sin(angle_ik))
        for i_idx, i in enumerate(pq_indices):
            for k_idx, k in enumerate(pq_indices):
                if i == k: J22[i_idx, k_idx] = Q_calc[i] / V[i] - V[i] * B[i, i]
                else: angle_ik = delta[i] - delta[k]; J22[i_idx, k_idx] = V[i] * (G[i, k] * np.sin(angle_ik) - B[i, k] * np.cos(angle_ik))
        J = np.vstack([np.hstack([J11, J12]), np.hstack([J21, J22])])

        try:
            corrections = np.linalg.solve(J, mismatch_vector)
        except np.linalg.LinAlgError:
            return False, bus_data.copy(), iteration, 0.0

        d_delta = corrections[:len(non_slack_indices)]; d_V = corrections[len(non_slack_indices):]
        delta[non_slack_indices] += d_delta; V[pq_indices] += d_V

    final_iterations = iteration + 1 if is_converged else iteration
    if not is_converged: return False, bus_data.copy(), final_iterations, 0.0

    V_final_complex = V * np.exp(1j * delta)
    S_final_complex = V_final_complex * np.conj(y_bus @ V_final_complex)
    P_final_net_pu = S_final_complex.real; Q_final_net_pu = S_final_complex.imag

    result_bus_data = bus_data.copy()
    result_bus_data['V_final_pu'] = V; result_bus_data['Angle_final_deg'] = np.rad2deg(delta)
    
    pg_final = (P_final_net_pu * base_mva) + pd_per_bus.values
    qg_final = (Q_final_net_pu * base_mva) + qd_per_bus.values

    result_bus_data['Pg_final_MW'] = pg_final
    result_bus_data['Qg_final_MVAR'] = qg_final
    result_bus_data['Pd_final_MW'] = pd_per_bus.values
    result_bus_data['Qd_final_MVAR'] = qd_per_bus.values
    
    p_loss = pg_final.sum() - pd_per_bus.values.sum()
    
    return True, result_bus_data, final_iterations, p_loss