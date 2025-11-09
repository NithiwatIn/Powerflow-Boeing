# simulation/usecases/mpg_disconnection_case.py

import pandas as pd
from tabulate import tabulate
from ..newtonrapson_loadflow import run_newton_raphson
from ..ybus_builder import build_ybus
from ..frequency_response import simulate_frequency_dynamics

def run(system_data: dict) -> tuple:
    output_string = ""
    results_dict = None

    try:
        output_string += "[1] Running initial load flow to determine pre-disturbance state...\n"
        
        BASE_MVA = system_data.get('config', {}).get('BaseMVA', 100.0)
        BASE_FREQ = system_data.get('config', {}).get('BaseFrequency', 50.0)
        
        ybus = build_ybus(system_data['buses'], system_data['lines'])
        
        initial_bus_data = system_data['buses'].copy()
        initial_gen_data = system_data['generators'].copy()
        initial_load_data = system_data['loads'].copy()

        converged, lf_results_df, iterations, losses = run_newton_raphson(
            bus_data=initial_bus_data,
            gen_data=initial_gen_data,
            load_data=initial_load_data,
            y_bus=ybus,
            base_mva=BASE_MVA
        )
        
        if not converged:
            raise RuntimeError("Initial load flow failed to converge. Cannot proceed with dynamic simulation.")

        # --- ส่วนที่เพิ่มเข้ามา: แสดงตาราง Load Flow เริ่มต้น ---
        output_string += f"   - Pre-disturbance state converged in {iterations} iterations.\n"
        output_string += "\n--- Pre-Disconnection Load Flow Summary ---\n"
        display_df = lf_results_df[[
            'BusID', 'Type', 'V_final_pu', 'Angle_final_deg', 
            'Pg_final_MW', 'Qg_final_MVAR', 'Pd_final_MW', 'Qd_final_MVAR'
        ]].copy()
        table_string = tabulate(display_df, headers='keys', tablefmt='grid', showindex=False, floatfmt=".4f")
        output_string += table_string + "\n\n"
        # --- จบส่วนที่เพิ่มเข้ามา ---

        slack_bus_row = lf_results_df[lf_results_df['Type'] == 1]
        if slack_bus_row.empty:
            raise ValueError("No Slack Bus (Type 1) found to represent the main grid connection.")
        
        power_from_mpg_mw = slack_bus_row['Pg_final_MW'].iloc[0]
        
        output_string += f"   - Power imported from Main Grid: {power_from_mpg_mw:.4f} MW\n"
        
        power_imbalance = power_from_mpg_mw if power_from_mpg_mw > 0 else 0
        if power_imbalance == 0:
            output_string += "   - System is not importing power. No under-frequency event expected.\n"

        output_string += "\n[2] Running dynamic frequency simulation (Primary Response)...\n"
        
        online_gens = system_data['generators'][system_data['generators']['Status'] == 1]
        
        freq_df = simulate_frequency_dynamics(
            online_generators=online_gens,
            power_imbalance_mw=power_imbalance,
            base_mva=BASE_MVA,
            base_freq=BASE_FREQ
        )
        
        nadir_freq = freq_df['frequency_hz'].min()
        settling_freq = freq_df['frequency_hz'].iloc[-1]
        
        output_string += "   - Dynamic simulation complete.\n"
        output_string += f"   - Frequency Nadir (จุดต่ำสุด): {nadir_freq:.4f} Hz\n"
        output_string += f"   - Settling Frequency (ความถี่คงตัวใหม่): {settling_freq:.4f} Hz\n"
        
        results_dict = {
            "dynamic_freq_df": freq_df
        }
        print(online_gens)
        print(power_imbalance)

    except Exception as e:
        output_string += f"\n--- AN ERROR OCCURRED IN USE CASE ---\n{str(e)}\n"
        
    return output_string, results_dict