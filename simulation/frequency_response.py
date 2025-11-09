# microgrid_project/simulation/dynamics/frequency_response.py

import numpy as np
import pandas as pd

def simulate_frequency_dynamics(online_generators: pd.DataFrame, power_imbalance_mw: float, 
                                base_mva: float, base_freq: float = 50.0, 
                                sim_duration_s: float = 10.0, dt: float = 0.01) -> pd.DataFrame:
    """
    จำลองการตอบสนองความถี่เบื้องต้น (Primary Frequency Response)
    โดยใช้ Aggregate Swing Equation และ Droop Control
    """
    
    # 1. คำนวณพารามิเตอร์รวมของระบบ (System Aggregate Parameters)
    # ค่าแรงเฉื่อยรวม (Equivalent Inertia)
    H_eq = (online_generators['Inertia_H'] * online_generators['Pmax_MW']).sum() / base_mva
    
    # ค่า Droop รวม (Equivalent Droop)
    # R_eq = 1 / sum(1/R_i)
    inv_R_sum = (1 / online_generators['Droop_R']).sum()
    R_eq = 1 / inv_R_sum if inv_R_sum > 0 else np.inf

    # Power imbalance in per-unit
    delta_P_L_pu = power_imbalance_mw / base_mva
    
    # 2. เริ่มการจำลองแบบวนซ้ำ (Euler Integration)
    timesteps = np.arange(0, sim_duration_s, dt)
    omega_pu = 1.0  # ความเร็วเชิงมุมเริ่มต้น (p.u.)
    
    results = []
    
    for t in timesteps:
        # ณ เวลา t=0, Governor ยังไม่ทำงาน, delta_P_m = 0
        # หลังจากนั้น, Governor จะปรับกำลังกลตาม droop
        if t == 0:
            delta_P_m_pu = 0.0
        else:
            # การตอบสนองของ Droop: dPm = -1/R * dOmega
            delta_P_m_pu = (-1 / R_eq) * (omega_pu - 1.0)

        # Aggregate Swing Equation: d(omega)/dt = (1 / 2H) * (dPm - dPe)
        d_omega_dt = (1 / (2 * H_eq)) * (delta_P_m_pu - delta_P_L_pu)
        
        # Update state for next step
        omega_pu += d_omega_dt * dt
        frequency_hz = omega_pu * base_freq
        
        results.append({'time_s': t, 'frequency_hz': frequency_hz})
        
    return pd.DataFrame(results)