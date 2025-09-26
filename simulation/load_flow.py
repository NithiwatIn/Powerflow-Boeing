# microgrid_project/simulation/load_flow.py

import numpy as np
import pandas as pd

def run_newton_raphson(bus_data: pd.DataFrame, gen_data: pd.DataFrame, load_data: pd.DataFrame, 
                         y_bus: np.ndarray, base_mva: float = 100.0, 
                         max_iter: int = 2000, tolerance: float = 1e-5):
    """
    คำนวณ Power Flow ด้วยวิธี Newton-Raphson
    **เวอร์ชันนี้ใช้แนวคิด Distributed Slack Bus และเคารพคุณสมบัติของบัสในการแสดงผล**
    """
    
    # --- 1. การเตรียมข้อมูลและตัวแปรเริ่มต้น ---
    G = y_bus.real
    B = y_bus.imag
    bus_types = bus_data['Type'].values
    slack_bus_idx = np.where(bus_types == 1)[0]
    pv_bus_indices = np.where(bus_types == 2)[0]
    pq_bus_indices = np.where(bus_types == 3)[0]
    non_slack_indices = np.where(bus_types != 1)[0]
    pq_pv_indices = np.concatenate([pv_bus_indices, pq_bus_indices])
    pq_pv_indices = np.sort(pq_pv_indices)
    pq_indices = np.sort(pq_bus_indices)
    num_buses = len(bus_data)
    V = bus_data['V_init'].values.copy()
    delta = np.deg2rad(bus_data['Angle_init'].values.copy())

    # --- 2. คำนวณกำลังไฟฟ้าที่กำหนด (Scheduled Power) โดยใช้หลักการ Load Sharing ---
    adjusted_gen_data = gen_data[gen_data['Status'] == 1].copy()
    total_pd_mw = load_data['Pd_MW'].sum()
    total_pg_initial_mw = adjusted_gen_data['Pg_MW'].sum()
    initial_imbalance_mw = total_pd_mw - total_pg_initial_mw
    print(f"\nSimulation Info: Initial Power Imbalance = {initial_imbalance_mw:.4f} MW (Load - Gen)")
    pf_sum = adjusted_gen_data['ParticipationFactor'].sum()
    if not np.isclose(pf_sum, 1.0):
        print(f"  [Warning] Sum of Participation Factors is {pf_sum:.4f}, not 1.0. Results may be scaled.")
    if np.isclose(pf_sum, 0.0):
        raise ValueError("Sum of Participation Factors is zero. Cannot distribute power imbalance.")
    adjustment = initial_imbalance_mw * (adjusted_gen_data['ParticipationFactor'] / pf_sum)
    adjusted_gen_data['Pg_MW'] += adjustment
    print("  - Adjusting generator outputs based on Participation Factors...")
    
    P_sch = np.zeros(num_buses)
    Q_sch = np.zeros(num_buses)
    for i in range(num_buses):
        bus_id = i + 1
        pg = adjusted_gen_data.loc[adjusted_gen_data['BusID'] == bus_id, 'Pg_MW'].sum()
        pd = load_data.loc[load_data['BusID'] == bus_id, 'Pd_MW'].sum()
        P_sch[i] = (pg - pd) / base_mva
        qg = adjusted_gen_data.loc[adjusted_gen_data['BusID'] == bus_id, 'Qg_MVAR'].sum()
        qd = load_data.loc[load_data['BusID'] == bus_id, 'Qd_MVAR'].sum()
        Q_sch[i] = (qg - qd) / base_mva

    # --- 3. เริ่มการคำนวณแบบวนซ้ำ (Iteration Loop) ---
    is_converged = False
    for iteration in range(max_iter):
        P_calc = np.zeros(num_buses)
        Q_calc = np.zeros(num_buses)
        for i in range(num_buses):
            for k in range(num_buses):
                angle_ik = delta[i] - delta[k]
                P_calc[i] += V[i] * V[k] * (G[i, k] * np.cos(angle_ik) + B[i, k] * np.sin(angle_ik))
                Q_calc[i] += V[i] * V[k] * (G[i, k] * np.sin(angle_ik) - B[i, k] * np.cos(angle_ik))
        delta_P = P_sch - P_calc
        delta_Q = Q_sch - Q_calc
        mismatch_P = delta_P[pq_pv_indices]
        mismatch_Q = delta_Q[pq_indices]
        mismatch_vector = np.concatenate([mismatch_P, mismatch_Q])
        if np.max(np.abs(mismatch_vector)) < tolerance:
            is_converged = True
            print(f"Newton-Raphson converged in {iteration+1} iterations.")
            break
        J11 = np.zeros((len(pq_pv_indices), len(pq_pv_indices)))
        J12 = np.zeros((len(pq_pv_indices), len(pq_indices)))
        J21 = np.zeros((len(pq_indices), len(pq_pv_indices)))
        J22 = np.zeros((len(pq_indices), len(pq_indices)))
        # ... (ส่วนการสร้าง Jacobian matrix เหมือนเดิม) ...
        for i_idx, i in enumerate(pq_pv_indices):
            for k_idx, k in enumerate(pq_pv_indices):
                if i == k: J11[i_idx, k_idx] = -Q_calc[i] - V[i]**2 * B[i, i]
                else: angle_ik = delta[i] - delta[k]; J11[i_idx, k_idx] = V[i] * V[k] * (G[i, k] * np.sin(angle_ik) - B[i, k] * np.cos(angle_ik))
        for i_idx, i in enumerate(pq_indices):
            for k_idx, k in enumerate(pq_pv_indices):
                if i == k: J21[i_idx, k_idx] = P_calc[i] - V[i]**2 * G[i, i]
                else: angle_ik = delta[i] - delta[k]; J21[i_idx, k_idx] = -V[i] * V[k] * (G[i, k] * np.cos(angle_ik) + B[i, k] * np.sin(angle_ik))
        for i_idx, i in enumerate(pq_pv_indices):
            for k_idx, k in enumerate(pq_indices):
                if i == k: J12[i_idx, k_idx] = P_calc[i]/V[i] + V[i] * G[i, i]
                else: angle_ik = delta[i] - delta[k]; J12[i_idx, k_idx] = V[i] * (G[i, k] * np.cos(angle_ik) + B[i, k] * np.sin(angle_ik))
        for i_idx, i in enumerate(pq_indices):
            for k_idx, k in enumerate(pq_indices):
                if i == k: J22[i_idx, k_idx] = Q_calc[i]/V[i] - V[i] * B[i, i]
                else: angle_ik = delta[i] - delta[k]; J22[i_idx, k_idx] = V[i] * (G[i, k] * np.sin(angle_ik) - B[i, k] * np.cos(angle_ik))
        J = np.vstack([np.hstack([J11, J12]), np.hstack([J21, J22])])
        try:
            corrections = np.linalg.solve(J, mismatch_vector)
        except np.linalg.LinAlgError:
            print("Error: Jacobian matrix is singular. Cannot solve.")
            return False, bus_data, iteration, 0.0
        d_delta = corrections[:len(pq_pv_indices)]; d_V = corrections[len(pq_pv_indices):]
        delta[pq_pv_indices] += d_delta; V[pq_indices] += d_V

    # --- 4. สรุปผลลัพธ์หลังจากการคำนวณ ---
    if not is_converged:
        print(f"Newton-Raphson failed to converge after {max_iter} iterations.")

    final_p_net_pu = np.zeros(num_buses); final_q_net_pu = np.zeros(num_buses)
    for i in range(num_buses):
        for k in range(num_buses):
            angle_ik = delta[i] - delta[k]
            final_p_net_pu[i] += V[i] * V[k] * (G[i, k] * np.cos(angle_ik) + B[i, k] * np.sin(angle_ik))
            final_q_net_pu[i] += V[i] * V[k] * (G[i, k] * np.sin(angle_ik) - B[i, k] * np.cos(angle_ik))
    
    result_bus_data = bus_data.copy()
    result_bus_data['V_final_pu'] = V
    result_bus_data['Angle_final_deg'] = np.rad2deg(delta)
    result_bus_data['Pg_final_MW'] = 0.0; result_bus_data['Qg_final_MVAR'] = 0.0
    result_bus_data['Pd_final_MW'] = 0.0; result_bus_data['Qd_final_MVAR'] = 0.0
    
    gen_bus_ids = gen_data['BusID'].unique()

    for i in range(num_buses):
        bus_id = i + 1
        bus_type = result_bus_data.loc[i, 'Type']
        
        pd_on_bus = load_data.loc[load_data['BusID'] == bus_id, 'Pd_MW'].sum()
        qd_on_bus = load_data.loc[load_data['BusID'] == bus_id, 'Qd_MVAR'].sum()
        result_bus_data.loc[i, 'Pd_final_MW'] = pd_on_bus
        result_bus_data.loc[i, 'Qd_final_MVAR'] = qd_on_bus

        is_gen_bus = bus_id in gen_bus_ids
        if is_gen_bus:
            p_net_mw = final_p_net_pu[i] * base_mva
            q_net_mvar = final_q_net_pu[i] * base_mva
            
            if bus_type == 1: # Slack Bus
                # Pg ของ Slack Bus เป็นตัวแปรตาม, ต้องคำนวณจากผลลัพธ์
                result_bus_data.loc[i, 'Pg_final_MW'] = p_net_mw + pd_on_bus
            elif bus_type == 2: # PV Bus
                # Pg ของ PV Bus เป็นค่าคงที่, ให้ใช้ค่าที่กำหนดไว้ใน adjusted_gen_data
                pg_scheduled = adjusted_gen_data.loc[adjusted_gen_data['BusID'] == bus_id, 'Pg_MW'].sum()
                result_bus_data.loc[i, 'Pg_final_MW'] = pg_scheduled
            
            # Qg ของทั้ง Slack และ PV Bus เป็นตัวแปรตาม, ต้องคำนวณจากผลลัพธ์
            result_bus_data.loc[i, 'Qg_final_MVAR'] = q_net_mvar + qd_on_bus
        else: # PQ Bus
            result_bus_data.loc[i, 'Pg_final_MW'] = 0.0
            result_bus_data.loc[i, 'Qg_final_MVAR'] = 0.0

    total_gen_final = result_bus_data['Pg_final_MW'].sum()
    total_load_final = result_bus_data['Pd_final_MW'].sum()
    p_loss = total_gen_final - total_load_final
    
    final_iterations = iteration + 1 if is_converged else iteration
    return is_converged, result_bus_data, final_iterations, p_loss