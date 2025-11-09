# microgrid_project/simulation/usecases/continuous_loadflow_case.py

import pandas as pd
import numpy as np
from ..newtonrapson_loadflow import run_newton_raphson
from ..ybus_builder import build_ybus

def run(system_data: dict) -> tuple:
    output_string = ""
    results_dict = None
    
    try:
        output_string += "[1] Preparing for time-series simulation...\n"
        if system_data.get('load_profile') is None:
            raise FileNotFoundError("load_profile_pattern.csv not found in 'data/' directory.")
        
        config = system_data.get('config', {})
        BASE_MVA = config.get('BaseMVA', 100.0)
        
        buses = system_data['buses']
        lines = system_data['lines']
        initial_gens = system_data['generators']
        initial_loads = system_data['loads']
        load_profile = system_data['load_profile']['pattern_1'].tolist()
        num_steps = len(load_profile)
        
        
        initial_loads['pf'] = (initial_loads['Pd_MW'] / 
                               ((initial_loads['Pd_MW']**2 + initial_loads['Qd_MVAR']**2)**0.5)).fillna(0.9)

        ybus_matrix = build_ybus(buses, lines)
        time_index = pd.to_datetime("00:00", format='%H:%M') + pd.to_timedelta(pd.Series(range(num_steps)) * 15, unit='m')
        
        output_string += f"[2] Running simulation for {num_steps} time steps...\n"
        all_results_dataframes = []
        
        for step, multiplier in enumerate(load_profile):
            current_loads = initial_loads.copy()
            current_loads['Pd_MW'] = initial_loads['Pd_MW'] * multiplier
            current_loads['Qd_MVAR'] = current_loads['Pd_MW'] * ((1 / current_loads['pf']**2) - 1)**0.5
            total_demand = current_loads['Pd_MW'].sum()
            
            dispatched_gens = initial_gens.copy()

            # --- ส่วนที่เพิ่มเข้ามา: PF Dispatch Logic ---
            active_gens = dispatched_gens[dispatched_gens['Status'] == 1].copy()
            pg_total_initial = active_gens['Pg_MW'].sum()
            initial_mismatch = total_demand - pg_total_initial
            
            slack_bus_id = buses[buses['Type'] == 1]['BusID'].iloc[0]
            participating_gens_mask = (active_gens['BusID'] != slack_bus_id)
            
            if participating_gens_mask.any():
                participating_gens = active_gens[participating_gens_mask]
                pf_sum = participating_gens['ParticipationFactor'].sum()
                if pf_sum > 1e-6:
                    adjustment = initial_mismatch * (participating_gens['ParticipationFactor'] / pf_sum)
                    active_gens.loc[participating_gens_mask, 'Pg_MW'] += adjustment
                    active_gens['Pg_MW'] = active_gens.apply(lambda r: np.clip(r['Pg_MW'], r['Pmin_MW'], r['Pmax_MW']), axis=1)
                    
                    dispatched_gens.set_index('GenID', inplace=True)
                    dispatched_gens.update(active_gens.set_index('GenID')['Pg_MW'])
                    dispatched_gens.reset_index(inplace=True)
            # --- จบส่วน PF Dispatch Logic ---

            converged, results_df, _, losses = run_newton_raphson(
                bus_data=buses,
                gen_data=dispatched_gens,
                load_data=current_loads,
                y_bus=ybus_matrix,
                base_mva=BASE_MVA
            )
            
            if converged:
                # Post-processing clamp for slack bus display
                slack_bus_row = results_df[results_df['Type'] == 1]
                if not slack_bus_row.empty:
                    slack_bus_id_current = slack_bus_row['BusID'].iloc[0]
                    gen_at_slack = dispatched_gens[dispatched_gens['BusID'] == slack_bus_id_current]
                    if not gen_at_slack.empty:
                        pmax_slack = gen_at_slack['Pmax_MW'].iloc[0]
                        slack_row_index = slack_bus_row.index[0]
                        current_pg = results_df.loc[slack_row_index, 'Pg_final_MW']
                        if current_pg > pmax_slack:
                            results_df.loc[slack_row_index, 'Pg_final_MW'] = pmax_slack

                results_df['Frequency_Hz'] = 50.0
                results_df['time_step'] = step
                all_results_dataframes.append(results_df)
            else:
                output_string += f"\n[ERROR] Load flow did not converge at time step {step} ({time_index[step].strftime('%H:%M')})."
                if all_results_dataframes:
                    last_good_result = all_results_dataframes[-1].copy()
                    last_good_result['time_step'] = step
                    all_results_dataframes.append(last_good_result)
                else:
                    raise RuntimeError(f"NR Converge Failed at step {step}")
        
        if not all_results_dataframes:
            raise RuntimeError("Simulation failed to produce any results.")

        output_string += "\n[3] Consolidating and formatting results...\n"
        full_df = pd.concat(all_results_dataframes, ignore_index=True)
        
        time_map = {step: time for step, time in enumerate(time_index)}
        full_df['datetime'] = full_df['time_step'].map(time_map)
        
        gen_info = system_data['generators'][['GenID', 'BusID']]
        results_with_gen_id = pd.merge(full_df, gen_info, on='BusID', how='left')
        online_gens_df = results_with_gen_id.dropna(subset=['GenID'])
        pivoted_gens = online_gens_df.pivot_table(index='datetime', columns='GenID', values='Pg_final_MW', aggfunc='sum').fillna(0)
        total_load_mw = full_df.groupby('datetime')['Pd_final_MW'].sum()
        
        results_dict = {
            "full_df": full_df,
            "summary_data": {
                "total_load_mw": total_load_mw,
                "pivoted_gens_mw": pivoted_gens,
            }
        }
        output_string += "\nContinuous Load Flow Simulation Completed."
        
    except Exception as e:
        import traceback
        output_string += f"\n--- AN ERROR OCCURRED IN '{run.__name__}' USE CASE ---\n"
        output_string += f"Error Type: {type(e).__name__}\n"
        output_string += f"Error Message: {e}\n"
        output_string += "--- Traceback ---\n"
        output_string += traceback.format_exc()
        results_dict = None

    return output_string, results_dict