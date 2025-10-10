# microgrid_project/simulation/use_cases/continuous_load_flow_case.py

import pandas as pd
from ..newtonrapson_loadflow import run_newton_raphson
from ..ybus_builder import build_ybus

def run(system_data: dict) -> tuple:
    """
    รัน Use Case: Continuous Load Flow
    - จำลองการทำงาน 24 ชม. (96 steps) ตาม Load Profile
    - คืนค่า (string_summary, results_dict) ที่มีข้อมูลเวลา
    """
    output_string = ""
    results_dict = None
    BASE_MVA = 100.0

    try:
        # --- 1. เตรียมข้อมูล ---
        output_string += "[1] Preparing for time-series simulation...\n"
        if system_data.get('load_profile') is None:
            raise FileNotFoundError("load_profile_pattern.csv not found in 'data/' directory.")
        
        load_multipliers = system_data['load_profile']['pattern_1'].tolist()
        num_steps = len(load_multipliers)
        
        if num_steps != 96:
            output_string += f"[WARNING] Load profile has {num_steps} points, not 96. Running for available points.\n"
        
        initial_loads = system_data['loads'].copy()
        initial_loads['pf'] = (initial_loads['Pd_MW'] / 
                              ((initial_loads['Pd_MW']**2 + initial_loads['Qd_MVAR']**2)**0.5)).fillna(0.9)

        ybus_matrix = build_ybus(system_data['buses'], system_data['lines'])
        
        # --- จุดที่แก้ไข: สร้าง Index เวลา ---
        # สร้าง list ของเวลาตลอด 24 ชม. ห่างกันทุก 15 นาที
        time_index = pd.to_datetime("00:00", format='%H:%M') + pd.to_timedelta(pd.Series(range(num_steps)) * 15, unit='m')
        
        # --- 2. เริ่มการจำลองแบบวนซ้ำ ---
        output_string += f"[2] Running simulation for {num_steps} time steps...\n"
        all_results = []
        
        for step, multiplier in enumerate(load_multipliers):
            current_loads = initial_loads.copy()
            current_loads['Pd_MW'] = initial_loads['Pd_MW'] * multiplier
            current_loads['Qd_MVAR'] = current_loads['Pd_MW'] * ((1 / current_loads['pf']**2) - 1)**0.5
            
            converged, results_df, _, _ = run_newton_raphson(
                bus_data=system_data['buses'],
                gen_data=system_data['generators'],
                load_data=current_loads,
                y_bus=ybus_matrix,
                base_mva=BASE_MVA,
                max_iter=30
            )
            
            if converged:
                results_df['time'] = time_index[step].strftime('%H:%M') # เพิ่มคอลัมน์ 'time'
                results_df['time_step'] = step
                all_results.append(results_df)
            else:
                output_string += f"    - Warning: Did not converge at time step {step} ({time_index[step].strftime('%H:%M')})\n"
        
        if not all_results:
            output_string += "\nSimulation failed to produce any results.\n"
            return output_string, None

        # --- 3. จัดการและสรุปผลลัพธ์ ---
        output_string += "[3] Consolidating and formatting results...\n"
        full_results_df = pd.concat(all_results, ignore_index=True)
        
        # เตรียมข้อมูลสำหรับกราฟ (ใช้ time_index ที่เป็น datetime object จริงๆ เพื่อให้แกน x สวยงาม)
        total_load_mw = full_results_df.groupby('time_step')['Pd_final_MW'].sum()
        total_load_mw.index = time_index[:len(total_load_mw)]

        pivoted_v = full_results_df.pivot(index='time_step', columns='BusID', values='V_final_pu')
        pivoted_v.index = time_index[:len(pivoted_v)]

        pivoted_angle = full_results_df.pivot(index='time_step', columns='BusID', values='Angle_final_deg')
        pivoted_angle.index = time_index[:len(pivoted_angle)]

        results_dict = {
            "full_df": full_results_df,
            "plot_data": {
                "total_load_mw": total_load_mw,
                "voltages": pivoted_v,
                "angles": pivoted_angle
            }
        }
        
        output_string += "\n--- Simulation Summary ---\n"
        output_string += f"Successfully completed {len(all_results)} of {num_steps} time steps.\n"
        peak_time = total_load_mw.idxmax().strftime('%H:%M')
        output_string += f"Peak Load: {total_load_mw.max():.2f} MW at {peak_time}\n"
        output_string += f"Min Voltage: {pivoted_v.min().min():.4f} p.u.\n"
        output_string += "-" * 80 + "\n"
        
    except Exception as e:
        output_string += f"\n--- AN ERROR OCCURRED IN USE CASE ---\n{str(e)}\n"

    return output_string, results_dict