# simulation/usecases/load_shedding_percentage_case.py

import pandas as pd
import numpy as np
import random
from ..newtonrapson_loadflow import run_newton_raphson
from ..ybus_builder import build_ybus

def _get_disconnection_step(config: dict, num_steps: int) -> int:
    disconnect_value = config.get('Disconnecting_Time', 99)
    try: disconnect_value = float(disconnect_value)
    except (ValueError, TypeError): raise TypeError(f"Invalid 'Disconnecting_Time' value '{disconnect_value}'. Must be a number.")
    if disconnect_value == 99:
        step = random.randint(1, num_steps - 6); return step
    if disconnect_value == int(disconnect_value): step = int(disconnect_value)
    else:
        hour = int(disconnect_value); minute = int(round((disconnect_value * 100) % 100))
        total_minutes = hour * 60 + minute; step = total_minutes // 15
    if not (0 <= step < num_steps):
        raise ValueError(f"Invalid 'Disconnecting_Time' ({disconnect_value}) results in an out-of-bounds step ({step}).")
    return step

def _run_dispatch_logic(total_demand, active_gens, microgrid_pmax, R_sys_hz_mw, BASE_FREQ):
    final_imbalance = 0; final_freq = BASE_FREQ
    if total_demand > microgrid_pmax:
        final_imbalance = total_demand - microgrid_pmax
        active_gens['Pg_MW'] = active_gens['Pmax_MW']
        delta_f = -R_sys_hz_mw * final_imbalance
        final_freq = BASE_FREQ + delta_f
    else:
        pf_sum = active_gens['ParticipationFactor'].sum()
        if pf_sum > 0: active_gens['Pg_MW'] = (total_demand / pf_sum) * active_gens['ParticipationFactor']
        else: active_gens['Pg_MW'] = total_demand / len(active_gens) if len(active_gens) > 0 else 0
        active_gens['Pg_MW'] = active_gens.apply(lambda r: np.clip(r['Pg_MW'], r['Pmin_MW'], r['Pmax_MW']), axis=1)
    total_pg_actual = active_gens['Pg_MW'].sum()
    return total_pg_actual, final_imbalance, final_freq, active_gens

def run(system_data: dict) -> tuple:
    output_string = ""
    results_dict = None
    try:
        config = system_data.get('config', {}); load_profile = system_data['load_profile']['pattern_1']; num_steps = len(load_profile)
        disconnection_time_step = _get_disconnection_step(config, num_steps)
        BASE_MVA = config.get('BaseMVA', 100.0); BASE_FREQ = config.get('BaseFrequency', 50.0)
        FREQ_THRESHOLD = 49.7 
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
        R_sys_hz_mw = 0
        if not online_dg.empty:
            inv_r_sum_hz_mw = 0
            for _, gen in online_dg.iterrows():
                if gen['Pmax_MW'] > 0: inv_r_sum_hz_mw += 1 / ((gen['Droop_R'] * BASE_FREQ) / gen['Pmax_MW'])
            if inv_r_sum_hz_mw > 0: R_sys_hz_mw = 1 / inv_r_sum_hz_mw
        time_index = pd.to_datetime("00:00", format='%H:%M') + pd.to_timedelta(pd.Series(range(num_steps)) * 15, unit='m')
        all_results_dataframes = []; summary_data_list = []
        all_shed_loads_list = [] 

        for i in range(num_steps):
            current_loads_base = initial_loads.copy()
            current_loads_base['Pd_MW'] *= load_profile[i]; current_loads_base['Qd_MVAR'] *= load_profile[i]
            total_demand_before = current_loads_base['Pd_MW'].sum()
            current_buses = buses.copy(); final_dispatch_gens = initial_gens.copy()
            is_islanding = (i >= disconnection_time_step)

            loads_after_shedding = current_loads_base.copy()
            total_demand_after = total_demand_before
            mw_shed = 0
            
            # --- ส่วนที่แก้ไข: แยกตรรกะก่อนและหลัง Disconnect ---
            if is_islanding:
                current_buses.loc[current_buses['BusID'] == mpg_bus_id, 'Type'] = 3
                current_buses.loc[current_buses['BusID'] == new_slack_bus_id, 'Type'] = 1
                
                active_dgs_base = final_dispatch_gens[(final_dispatch_gens['BusID'] != mpg_bus_id) & (final_dispatch_gens['Status'] == 1)].copy()
                total_pg_before, imbalance_before, freq_before, dispatched_gens_before = _run_dispatch_logic(
                    total_demand_before, active_dgs_base, microgrid_pmax_total, R_sys_hz_mw, BASE_FREQ
                )
                
                freq_after = freq_before
                total_pg_after = total_pg_before
                dispatched_gens_after = dispatched_gens_before
                shed_percentages = {}

                if freq_before < FREQ_THRESHOLD:
                    sheddable_loads = loads_after_shedding[(loads_after_shedding['Status'] == 1) & (loads_after_shedding['Pd_MW'] > 0.001)].copy()
                    while freq_after < FREQ_THRESHOLD and not sheddable_loads.empty:
                        min_priority = sheddable_loads['Priority'].min()
                        loads_with_min_priority = sheddable_loads[sheddable_loads['Priority'] == min_priority]
                        load_to_cut_id = loads_with_min_priority['Pd_MW'].idxmin()
                        
                        current_shed_percent = shed_percentages.get(load_to_cut_id, 0.0)
                        new_shed_percent = min(current_shed_percent + 0.1, 1.0)
                        
                        original_load_row = current_loads_base.loc[load_to_cut_id]
                        pd_original = original_load_row['Pd_MW']; qd_original = original_load_row['Qd_MVAR']
                        
                        pd_new = pd_original * (1.0 - new_shed_percent)
                        qd_new = qd_original * (1.0 - new_shed_percent)
                        
                        loads_after_shedding.loc[load_to_cut_id, 'Pd_MW'] = pd_new
                        loads_after_shedding.loc[load_to_cut_id, 'Qd_MVAR'] = qd_new
                        shed_percentages[load_to_cut_id] = new_shed_percent
                        
                        if new_shed_percent >= 1.0:
                            sheddable_loads = sheddable_loads.drop(load_to_cut_id)
                            
                        total_demand_after = loads_after_shedding['Pd_MW'].sum()
                        mw_shed = total_demand_before - total_demand_after
                        
                        active_dgs_after = final_dispatch_gens[(final_dispatch_gens['BusID'] != mpg_bus_id) & (final_dispatch_gens['Status'] == 1)].copy()
                        total_pg_after, _, freq_after, dispatched_gens_after = _run_dispatch_logic(
                            total_demand_after, active_dgs_after, microgrid_pmax_total, R_sys_hz_mw, BASE_FREQ
                        )
                
                for load_id, percent in shed_percentages.items():
                    shed_load_row = initial_loads.loc[load_id]
                    mw_shed_total = shed_load_row['Pd_MW'] * load_profile[i] * percent
                    mvar_shed_total = shed_load_row['Qd_MVAR'] * load_profile[i] * percent
                    all_shed_loads_list.append({
                        'datetime': time_index[i], 'BusID': shed_load_row['BusID'],
                        'Priority': shed_load_row['Priority'], 'Shed_Percent': percent * 100,
                        'MW_Shed': mw_shed_total, 'MVAR_Shed': mvar_shed_total
                    })

            else:
                # ก่อน Disconnect, ความถี่เป็น 50 Hz เสมอ
                freq_before = BASE_FREQ
                freq_after = BASE_FREQ
                total_pg_after = total_demand_before
                dispatched_gens_after = final_dispatch_gens[(final_dispatch_gens['BusID'] != mpg_bus_id) & (final_dispatch_gens['Status'] == 1)].copy()
            # --- จบส่วนที่แก้ไข ---

            final_dispatch_gens.set_index('GenID', inplace=True)
            if 'Pg_MW' in dispatched_gens_after.columns:
                final_dispatch_gens.update(dispatched_gens_after.set_index('GenID')['Pg_MW'])
            final_dispatch_gens.reset_index(inplace=True)
            final_dispatch_gens.loc[final_dispatch_gens['BusID'] == mpg_bus_id, 'Pg_MW'] = 0.0

            converged, results_df, _, _ = run_newton_raphson(
                bus_data=current_buses, gen_data=final_dispatch_gens, load_data=loads_after_shedding, 
                y_bus=ybus_matrix, base_mva=BASE_MVA, perform_pf_dispatch=(not is_islanding)
            )
            
            if converged:
                if not is_islanding: total_pg_after = results_df['Pg_final_MW'].sum()
                results_df['Frequency_Hz'] = freq_after; results_df['time_step'] = i
                all_results_dataframes.append(results_df)
                summary_data_list.append({
                    'datetime': time_index[i], 'freq_before': freq_before, 'freq_after': freq_after,
                    'load_before': total_demand_before, 'load_after': total_demand_after,
                    'gen_total': total_pg_after, 'mw_shed': mw_shed
                })
            else:
                raise RuntimeError(f"NR Converge Failed at step {i} ({time_index[i].strftime('%H:%M')})")
            
        if not all_results_dataframes: raise RuntimeError("Simulation failed to produce any results.")

        full_df = pd.concat(all_results_dataframes).reset_index(drop=True)
        time_map = {step: time for step, time in enumerate(time_index)}
        full_df['datetime'] = full_df['time_step'].map(time_map)
        summary_df = pd.DataFrame(summary_data_list).set_index('datetime')
        shed_loads_df = pd.DataFrame(all_shed_loads_list)
        
        results_dict = { 
            "full_df": full_df, 
            "shed_loads_df": shed_loads_df,
            "summary_data": { 
                "total_load_mw_before": summary_df['load_before'], "total_load_mw_after": summary_df['load_after'],
                "total_pg_mw": summary_df['gen_total'], "frequency_series_before": summary_df[['freq_before']],
                "frequency_series_after": summary_df[['freq_after']], "mw_shed_series": summary_df[['mw_shed']],
                "disconnection_time": time_index[disconnection_time_step], 
                "microgrid_pmax": microgrid_pmax_total, "freq_threshold": FREQ_THRESHOLD
            },
            "calculation_params": { "base_mva": BASE_MVA, "base_freq": BASE_FREQ, "online_dgs": online_dg }
        }
        output_string += "\nLoad Shedding (Percentage) Simulation Completed Successfully."
        
    except Exception as e:
        import traceback
        output_string = f"\n--- AN ERROR OCCURRED IN '{run.__name__}' USE CASE ---\n"; output_string += f"Error Type: {type(e).__name__}\n"; output_string += f"Error Message: {e}\n"; output_string += "--- Traceback ---\n"; output_string += traceback.format_exc(); results_dict = None
    return output_string, results_dict