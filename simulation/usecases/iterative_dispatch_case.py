# simulation/usecases/iterative_dispatch_case.py

import pandas as pd
import numpy as np
import random
from ..newtonrapson_loadflow import run_newton_raphson
from ..ybus_builder import build_ybus

def _get_disconnection_step(config: dict, num_steps: int) -> int:
    """
    แปลความหมายของ Disconnecting_Time จากไฟล์ config
    - 99: สุ่มเวลา
    - ทศนิยม (HH.MM): แปลงเป็น step
    - จำนวนเต็ม: ใช้เป็น step โดยตรง
    """
    disconnect_value = config.get('Disconnecting_Time', 99) # Default to random if not specified

    try:
        disconnect_value = float(disconnect_value)
    except (ValueError, TypeError):
        raise TypeError(f"Invalid 'Disconnecting_Time' value '{disconnect_value}'. Must be a number.")

    if disconnect_value == 99:
        # สุ่ม step โดยไม่เอา step แรกสุดและ 5 step สุดท้าย
        step = random.randint(1, num_steps - 6)
        print(f"   - Random disconnection time selected: Step {step}")
        return step
    
    # ตรวจสอบว่าเป็นจำนวนเต็มหรือทศนิยม
    if disconnect_value == int(disconnect_value): # Case: จำนวนเต็ม
        step = int(disconnect_value)
    else: # Case: ทศนิยม (HH.MM)
        hour = int(disconnect_value)
        minute = int(round((disconnect_value * 100) % 100))
        total_minutes = hour * 60 + minute
        step = total_minutes // 15
        print(f"   - Disconnection time {hour:02d}:{minute:02d} converted to Step {step}")
    
    # ตรวจสอบความถูกต้องของ step สุดท้าย
    if not (0 <= step < num_steps):
        raise ValueError(
            f"Invalid 'Disconnecting_Time' ({disconnect_value}) results in an out-of-bounds step ({step}). "
            f"Please choose a value that results in a step between 0 and {num_steps - 1}."
        )
    return step


def run(system_data: dict) -> tuple:
    output_string = ""
    results_dict = None
    try:
        config = system_data.get('config', {}); load_profile = system_data['load_profile']['pattern_1']; num_steps = len(load_profile)
        disconnection_time_step = _get_disconnection_step(config, num_steps)
        BASE_MVA = config.get('BaseMVA', 100.0); BASE_FREQ = config.get('BaseFrequency', 50.0)
        buses = system_data['buses']; lines = system_data['lines']
        initial_gens = system_data['generators']; initial_loads = system_data['loads']
        ybus_matrix = build_ybus(buses, lines)

        mpg_bus_id = buses[buses['Type'] == 1]['BusID'].iloc[0]
        microgrid_gens = initial_gens[initial_gens['BusID'] != mpg_bus_id]
        microgrid_pmax_total = microgrid_gens[microgrid_gens['Status'] == 1]['Pmax_MW'].sum()
        new_slack_gen = microgrid_gens.loc[microgrid_gens['Pmax_MW'].idxmax()] if not microgrid_gens.empty else None
        if new_slack_gen is None: raise ValueError("No microgrid generators found.")
        new_slack_bus_id = new_slack_gen['BusID']

        online_dg = microgrid_gens[microgrid_gens['Status'] == 1]
        inv_R_sum = (1 / online_dg['Droop_R']).sum()
        R_sys = (1 / inv_R_sum) if inv_R_sum > 0 else np.inf
        # --- ส่วนที่แก้ไข: แปลง R_sys เป็น Hz/MW ที่นี่เลย ---
        R_sys_hz_mw = 0
        if not online_dg.empty:
            inv_r_sum_hz_mw = 0
            for _, gen in online_dg.iterrows():
                if gen['Pmax_MW'] > 0:
                    inv_r_sum_hz_mw += 1 / ((gen['Droop_R'] * BASE_FREQ) / gen['Pmax_MW'])
            if inv_r_sum_hz_mw > 0:
                R_sys_hz_mw = 1 / inv_r_sum_hz_mw

        time_index = pd.to_datetime("00:00", format='%H:%M') + pd.to_timedelta(pd.Series(range(num_steps)) * 15, unit='m')
        all_results_dataframes = []; summary_data_list = []

        for i in range(num_steps):
            current_loads = initial_loads.copy()
            current_loads['Pd_MW'] *= load_profile[i]
            current_loads['Qd_MVAR'] *= load_profile[i]
            total_demand = current_loads['Pd_MW'].sum()

            current_buses = buses.copy()
            final_dispatch_gens = initial_gens.copy()
            is_islanding = (i >= disconnection_time_step)

            final_imbalance = 0; total_pg_actual = 0; final_freq = BASE_FREQ

            if is_islanding:
                current_buses.loc[current_buses['BusID'] == mpg_bus_id, 'Type'] = 3
                current_buses.loc[current_buses['BusID'] == new_slack_bus_id, 'Type'] = 1
                
                active_dgs = final_dispatch_gens[(final_dispatch_gens['BusID'] != mpg_bus_id) & (final_dispatch_gens['Status'] == 1)].copy()

                if total_demand > microgrid_pmax_total:
                    final_imbalance = total_demand - microgrid_pmax_total
                    active_dgs['Pg_MW'] = active_dgs['Pmax_MW']
                    delta_f = -R_sys * final_imbalance
                    final_freq = BASE_FREQ + delta_f
                else:
                    pf_sum = active_dgs['ParticipationFactor'].sum()
                    if pf_sum > 0: active_dgs['Pg_MW'] = (total_demand / pf_sum) * active_dgs['ParticipationFactor']
                    else: active_dgs['Pg_MW'] = total_demand / len(active_dgs) if len(active_dgs) > 0 else 0
                    active_dgs['Pg_MW'] = active_dgs.apply(lambda r: np.clip(r['Pg_MW'], r['Pmin_MW'], r['Pmax_MW']), axis=1)

                total_pg_actual = active_dgs['Pg_MW'].sum()
                
                pg_update_map = active_dgs.set_index('GenID')['Pg_MW']
                final_dispatch_gens['Pg_MW'] = final_dispatch_gens['GenID'].map(pg_update_map).fillna(final_dispatch_gens['Pg_MW'])
                final_dispatch_gens.loc[final_dispatch_gens['BusID'] == mpg_bus_id, 'Pg_MW'] = 0.0

            delta_f = 0
            if is_islanding and final_imbalance > 0:
                delta_f = -R_sys_hz_mw * final_imbalance
            final_freq = BASE_FREQ + delta_f

            converged, results_df, _, _ = run_newton_raphson(
                bus_data=current_buses, gen_data=final_dispatch_gens, load_data=current_loads, 
                y_bus=ybus_matrix, base_mva=BASE_MVA, 
                perform_pf_dispatch=(not is_islanding)
            )
            
            
            if converged:
                slack_bus_row = results_df[results_df['Type'] == 1]
            if not slack_bus_row.empty:
                slack_bus_id = slack_bus_row['BusID'].iloc[0]
                gen_at_slack = initial_gens[initial_gens['BusID'] == slack_bus_id]
                
                if not gen_at_slack.empty:
                    pmax_slack = gen_at_slack['Pmax_MW'].iloc[0]
                    slack_row_index = slack_bus_row.index[0]
                    
                    # Clamp Pg
                    current_pg = results_df.loc[slack_row_index, 'Pg_final_MW']
                    if current_pg > pmax_slack:
                        results_df.loc[slack_row_index, 'Pg_final_MW'] = pmax_slack
                if not is_islanding:
                    total_pg_actual = results_df['Pg_final_MW'].sum()
                results_df['Frequency_Hz'] = final_freq
                results_df['time_step'] = i
                all_results_dataframes.append(results_df)
                summary_data_list.append({
                    'datetime': time_index[i], 
                    'frequency': final_freq, 
                    'delta_f': delta_f,
                    'total_pg_actual': total_pg_actual,
                    'power_imbalance': final_imbalance
                })
            else:
                output_string += f"\n[CRITICAL ERROR] Load flow did not converge at time step {i} ({time_index[i].strftime('%H:%M')}). Halting simulation."
                raise RuntimeError(f"NR Converge Failed at step {i}")
            
        if not all_results_dataframes: raise RuntimeError("Simulation failed to produce any results.")

        # --- 3. CONSOLIDATE RESULTS ---
        full_df = pd.concat(all_results_dataframes).reset_index(drop=True)
        time_map = {step: time for step, time in enumerate(time_index)}
        full_df['datetime'] = full_df['time_step'].map(time_map)
        summary_df = pd.DataFrame(summary_data_list).set_index('datetime')
        total_load_mw = full_df.groupby('datetime')['Pd_final_MW'].sum()
        
        results_dict = { 
            "full_df": full_df, 
            "summary_data": { 
                "total_load_mw": total_load_mw, 
                "total_pg_mw": summary_df['total_pg_actual'], 
                "frequency_series": summary_df[['frequency']],
                "power_imbalance_series": summary_df[['power_imbalance']],
                "delta_f_series": summary_df[['delta_f']], # <--- ส่ง Delta_f
                "disconnection_time": time_index[disconnection_time_step], 
                "microgrid_pmax": microgrid_pmax_total 
            },
            "calculation_params": { 
                "base_mva": BASE_MVA, "base_freq": BASE_FREQ,
                "online_dgs": online_dg, "r_sys_hz_mw": R_sys_hz_mw # <--- ส่ง R_sys
            }
        }
        output_string += "\nIterative Dispatch Simulation Completed Successfully."
        
    except Exception as e:
        import traceback
        output_string = f"\n--- AN ERROR OCCURRED IN '{run.__name__}' USE CASE ---\n"
        output_string += f"Error Type: {type(e).__name__}\n"
        output_string += f"Error Message: {e}\n"
        output_string += "--- Traceback ---\n"
        output_string += traceback.format_exc()
        results_dict = None

    return output_string, results_dict