# microgrid_project/simulation/use_cases/mpg_disconnection_case.py

import pandas as pd
import numpy as np
import random
from ..newtonrapson_loadflow import run_newton_raphson
from ..ybus_builder import build_ybus

def _calculate_frequency(power_imbalance_mw, online_gens_df, base_mva, f_nominal):
    """
    คำนวณความถี่ของระบบในสภาวะคงตัวตาม Droop Control
    """
    if online_gens_df.empty or np.isclose(power_imbalance_mw, 0, atol=1e-4):
        return f_nominal

    # ตรวจสอบว่า Pmax_MW ไม่เป็นศูนย์เพื่อป้องกันการหารด้วยศูนย์
    online_gens_df = online_gens_df[online_gens_df['Pmax_MW'] > 0]
    if online_gens_df.empty:
        return f_nominal

    system_droop_values = online_gens_df['Droop_R'] * (base_mva / online_gens_df['Pmax_MW'])
    
    if system_droop_values.empty or np.isclose((1 / system_droop_values).sum(), 0):
        return f_nominal
        
    r_eq_pu = 1 / (1 / system_droop_values).sum()
    delta_p_pu = power_imbalance_mw / base_mva
    delta_f_pu = -delta_p_pu * r_eq_pu
    
    frequency = f_nominal * (1 + delta_f_pu)
    return frequency

def run(system_data: dict) -> tuple:
    output_string = ""
    results_dict = None
    BASE_MVA = 100.0
    
    try:
        # --- 1. เตรียมข้อมูลและพารามิเตอร์การจำลอง ---
        output_string += "[1] Preparing for Islanded Simulation with MPG Disconnection...\n"
        config = system_data.get('config', {})
        MPG_BUS = int(config.get('MPG_Bus', 0))
        DISCONNECT_TIME_STR = str(config.get('Disconnecting_Time', '99'))
        F_NOMINAL = float(config.get('Nominal_Frequency', 50.0))
        
        if MPG_BUS == 0: raise ValueError("MPG_Bus not defined in system_config.csv")

        load_multipliers = system_data['load_profile']['pattern_1'].tolist()
        num_steps = len(load_multipliers)
        
        if DISCONNECT_TIME_STR == '99':
            disconnect_step = random.randint(int(num_steps * 0.25), int(num_steps * 0.75))
        else:
            disconnect_step = int(float(DISCONNECT_TIME_STR) * 4)
        
        time_index = pd.to_datetime("00:00", format='%H:%M') + pd.to_timedelta(pd.Series(range(num_steps)) * 15, unit='m')
        output_string += f"MPG (Generator at Bus {MPG_BUS}) will be disconnected at step {disconnect_step} ({time_index[disconnect_step].strftime('%H:%M')})\n"
        
        initial_loads = system_data['loads'].copy()
        initial_loads['pf'] = (initial_loads['Pd_MW'] / ((initial_loads['Pd_MW']**2 + initial_loads['Qd_MVAR']**2)**0.5)).fillna(0.9)
        ybus_matrix = build_ybus(system_data['buses'], system_data['lines'])
        
        # --- 2. เริ่มการจำลองแบบวนซ้ำ ---
        output_string += f"[2] Running simulation for {num_steps} time steps (Islanded Mode)... \n"
        all_results = []
        
        for step, multiplier in enumerate(load_multipliers):
            gen_data_for_step = system_data['generators'].copy()
            current_loads = initial_loads.copy()
            current_loads['Pd_MW'] *= multiplier
            current_loads['Qd_MVAR'] *= multiplier

            # ตรวจสอบเงื่อนไข Disconnect
            if step >= disconnect_step:
                # ตั้งค่า Status ของ MPG Gen เป็น 0 (Offline)
                mpg_gen_idx = gen_data_for_step[gen_data_for_step['BusID'] == MPG_BUS].index
                gen_data_for_step.loc[mpg_gen_idx, 'Status'] = 0

            # --- รัน Load Flow แบบ Distributed Slack เสมอ ---
            # เราต้องกำหนด Angle Reference Bus ในทุกๆ รอบ
            # โดยเลือก Gen ที่ใหญ่ที่สุดที่ยัง "Online" อยู่
            online_gens_in_step = gen_data_for_step[gen_data_for_step['Status'] == 1]
            if online_gens_in_step.empty:
                output_string += f"    - Error at step {step}: No generators online! System blackout.\n"
                break # หยุดการจำลอง

            ref_bus_id = online_gens_in_step.nlargest(1, 'Pmax_MW')['BusID'].iloc[0]
            
            bus_data_for_step = system_data['buses'].copy()
            bus_data_for_step['Type'] = 3 # Reset all to PQ
            bus_data_for_step.loc[bus_data_for_step['BusID'] == ref_bus_id, 'Type'] = 1
            
            converged, results_df, _, losses = run_newton_raphson(
                bus_data=bus_data_for_step, gen_data=gen_data_for_step,
                load_data=current_loads, y_bus=ybus_matrix, base_mva=BASE_MVA)

            # คำนวณความถี่ในทุกๆ รอบ
            if converged:
                power_imbalance_for_freq = abs(losses)
                freq = _calculate_frequency(power_imbalance_for_freq, online_gens_in_step, BASE_MVA, F_NOMINAL)
                results_df['frequency_hz'] = freq
                results_df['time'] = time_index[step].strftime('%H:%M')
                all_results.append(results_df)
            else:
                output_string += f"    - Warning: Did not converge at step {step} ({time_index[step].strftime('%H:%M')})\n"
        
        if not all_results:
            output_string += "\nSimulation failed to produce any results.\n"
            return output_string, None

        # --- 3. จัดการและสรุปผลลัพธ์ ---
        output_string += "[3] Consolidating results...\n"
        full_results_df = pd.concat(all_results, ignore_index=True)
        
        full_results_df = full_results_df[['time'] + [col for col in full_results_df.columns if col != 'time']]
        
        total_load_mw = full_results_df.groupby('time')['Pd_final_MW'].sum()
        frequency_series = full_results_df.groupby('time')['frequency_hz'].mean()
        
        total_load_mw.index = pd.to_datetime(total_load_mw.index, format='%H:%M')
        frequency_series.index = pd.to_datetime(frequency_series.index, format='%H:%M')

        results_dict = {
            "full_df": full_results_df,
            "plot_data": {
                "total_load_mw": total_load_mw,
                "frequency": frequency_series
            }
        }
        
        output_string += "\n--- Simulation Summary ---\n"
        freq_min_time = frequency_series.idxmin().strftime('%H:%M')
        output_string += f"Lowest Frequency: {frequency_series.min():.4f} Hz at {freq_min_time}\n"
        freq_max_time = frequency_series.idxmax().strftime('%H:%M')
        output_string += f"Highest Frequency: {frequency_series.max():.4f} Hz at {freq_max_time}\n"
        output_string += "-" * 80 + "\n"
        
    except Exception as e:
        output_string += f"\n--- AN ERROR OCCURRED IN USE CASE ---\n{str(e)}\n"
        results_dict = None

    return output_string, results_dict