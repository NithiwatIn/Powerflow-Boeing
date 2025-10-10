# microgrid_project/simulation/newtonrapson_loadflow.py

import numpy as np
import pandas as pd

def run_newton_raphson(bus_data: pd.DataFrame, gen_data: pd.DataFrame, load_data: pd.DataFrame, 
                         y_bus: np.ndarray, base_mva: float = 100.0, 
                         max_iter: int = 20, tolerance: float = 1e-5):
    """
    คำนวณ Power Flow ด้วยวิธี Newton-Raphson
    **เวอร์ชันนี้ใช้แนวคิด Distributed Slack Bus สำหรับ Islanded System**
    """
    
    # --- 1. การเตรียมข้อมูลและตัวแปรเริ่มต้น ---
    G = y_bus.real
    B = y_bus.imag
    bus_types = bus_data['Type'].values
    pv_bus_indices = np.where(bus_types == 2)[0]
    pq_bus_indices = np.where(bus_types == 3)[0]
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
    
    pf_sum = adjusted_gen_data['ParticipationFactor'].sum()
    if not np.isclose(pf_sum, 0.0):
        if not np.isclose(pf_sum, 1.0):
            print(f"  [Warning] Sum of Participation Factors is {pf_sum:.4f}, not 1.0. Results will be scaled.")
        adjustment = initial_imbalance_mw * (adjusted_gen_data['ParticipationFactor'] / pf_sum)
        adjusted_gen_data['Pg_MW'] += adjustment
    elif not np.isclose(initial_imbalance_mw, 0):
        # กรณี PF_sum = 0 (โหมด Slack Bus ปกติ)
        # Imbalance ทั้งหมดจะไปที่ Slack Bus
        slack_bus_id = bus_data.loc[bus_data['Type'] == 1, 'BusID'].iloc[0]
        adjusted_gen_data.loc[adjusted_gen_data['BusID'] == slack_bus_id, 'Pg_MW'] += initial_imbalance_mw

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
            break

        # (ส่วนของการสร้าง Jacobian และแก้สมการเหมือนเดิม)
        # ...
        J11 = np.zeros((len(pq_pv_indices), len(pq_pv_indices)))
        J12 = np.zeros((len(pq_pv_indices), len(pq_indices)))
        J21 = np.zeros((len(pq_indices), len(pq_pv_indices)))
        J22 = np.zeros((len(pq_indices), len(pq_indices)))
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
            print("Error: Jacobian matrix is singular.")
            return False, bus_data, iteration, 0.0
        d_delta = corrections[:len(pq_pv_indices)]; d_V = corrections[len(pq_pv_indices):]
        delta[pq_pv_indices] += d_delta; V[pq_indices] += d_V

    # --- 4. สรุปผลลัพธ์หลังจากการคำนวณ ---
    final_iterations = iteration + 1 if is_converged else iteration
    if not is_converged:
        print(f"Newton-Raphson failed to converge after {final_iterations} iterations.")
        return False, bus_data, final_iterations, 0.0

    result_bus_data = bus_data.copy()
    result_bus_data['V_final_pu'] = V
    result_bus_data['Angle_final_deg'] = np.rad2deg(delta)
    result_bus_data['Pg_final_MW'] = 0.0
    result_bus_data['Qg_final_MVAR'] = 0.0
    result_bus_data['Pd_final_MW'] = 0.0
    result_bus_data['Qd_final_MVAR'] = 0.0
    
    # คำนวณ Net Injection สุดท้าย
    final_p_net_pu = np.zeros(num_buses)
    final_q_net_pu = np.zeros(num_buses)
    for i in range(num_buses):
        for k in range(num_buses):
            angle_ik = delta[i] - delta[k]
            final_p_net_pu[i] += V[i] * V[k] * (G[i, k] * np.cos(angle_ik) + B[i, k] * np.sin(angle_ik))
            final_q_net_pu[i] += V[i] * V[k] * (G[i, k] * np.sin(angle_ik) - B[i, k] * np.cos(angle_ik))

    # --- จุดที่แก้ไข: ตรรกะการกรอกผลลัพธ์ ---
    for i in range(num_buses):
        bus_id = i + 1
        
        # 1. กรอกค่า Load
        pd_on_bus = load_data.loc[load_data['BusID'] == bus_id, 'Pd_MW'].sum()
        qd_on_bus = load_data.loc[load_data['BusID'] == bus_id, 'Qd_MVAR'].sum()
        result_bus_data.loc[i, 'Pd_final_MW'] = pd_on_bus
        result_bus_data.loc[i, 'Qd_final_MVAR'] = qd_on_bus

        # 2. กรอกค่า Generation
        # 2.1 หาค่า Qg ที่คำนวณได้สำหรับทุกบัสที่มี Gen
        q_net_mvar = final_q_net_pu[i] * base_mva
        result_bus_data.loc[i, 'Qg_final_MVAR'] = q_net_mvar + qd_on_bus
        
        # 2.2 หาค่า Pg โดยดึงจาก adjusted_gen_data ที่คำนวณไว้แล้ว
        # adjusted_gen_data มีเฉพาะ Gen ที่ Online
        pg_on_bus = adjusted_gen_data.loc[adjusted_gen_data['BusID'] == bus_id, 'Pg_MW'].sum()
        result_bus_data.loc[i, 'Pg_final_MW'] = pg_on_bus
    
    # 3. คำนวณ Power Loss
    total_gen_final = result_bus_data['Pg_final_MW'].sum()
    total_load_final = result_bus_data['Pd_final_MW'].sum()
    p_loss = total_gen_final - total_load_final
    
    return is_converged, result_bus_data, final_iterations, p_loss