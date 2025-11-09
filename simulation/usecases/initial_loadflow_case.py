# simulation/usecases/initial_loadflow_case.py

import pandas as pd
import numpy as np
from tabulate import tabulate
from ..ybus_builder import build_ybus
from ..newtonrapson_loadflow import run_newton_raphson

def run(system_data: dict) -> tuple:
    output_string = ""
    results_dict = None
    try:
        # Create copies to avoid modifying original data
        buses = system_data['buses'].copy()
        gen_data = system_data['generators'].copy()
        load_data = system_data['loads'].copy()
        BASE_MVA = system_data.get('config', {}).get('BaseMVA', 100.0)

        # --- PF Dispatch Logic now lives here ---
        pd_total = load_data['Pd_MW'].sum()
        active_gens = gen_data[gen_data['Status'] == 1].copy()
        pg_total_initial = active_gens['Pg_MW'].sum()
        initial_mismatch = pd_total - pg_total_initial
        
        slack_bus_id = buses[buses['Type'] == 1]['BusID'].iloc[0]
        participating_gens_mask = (active_gens['BusID'] != slack_bus_id)
        
        if participating_gens_mask.any():
            participating_gens = active_gens[participating_gens_mask]
            pf_sum = participating_gens['ParticipationFactor'].sum()
            if pf_sum > 1e-6:
                adjustment = initial_mismatch * (participating_gens['ParticipationFactor'] / pf_sum)
                active_gens.loc[participating_gens_mask, 'Pg_MW'] += adjustment
                active_gens['Pg_MW'] = active_gens.apply(lambda r: np.clip(r['Pg_MW'], r['Pmin_MW'], r['Pmax_MW']), axis=1)
                gen_data.set_index('GenID', inplace=True)
                gen_data.update(active_gens.set_index('GenID')['Pg_MW'])
                gen_data.reset_index(inplace=True)

        output_string += "[1] Building Y-Bus Matrix...\n"
        ybus_matrix = build_ybus(buses, system_data['lines'])
        output_string += "     Y-Bus built successfully.\n\n"
        output_string += f"[2] Running Newton-Raphson Load Flow...\n"
        output_string += f"     (Power dispatch adjusted by Participation Factor)\n"
        output_string += f"     Using Base MVA: {BASE_MVA}\n\n"
        
        converged, results_df, iterations, losses = run_newton_raphson(
            bus_data=buses, gen_data=gen_data, load_data=load_data,
            y_bus=ybus_matrix, base_mva=BASE_MVA
        )
        
        if converged:
            results_dict = {'full_df': results_df}
            output_string += f"Status: Converged in {iterations} iterations.\n"
            output_string += f"System Losses (P_loss): {losses:.4f} MW\n"
            output_string += "\n--- Final Bus Results Summary ---\n"
            display_df = results_df[['BusID', 'Type', 'V_final_pu', 'Angle_final_deg', 'Pg_final_MW', 'Qg_final_MVAR', 'Pd_final_MW', 'Qd_final_MVAR']].copy()
            output_string += tabulate(display_df, headers='keys', tablefmt='simple_outline', showindex=False, floatfmt=".4f") + "\n"
        else:
            output_string += f"Status: Did not converge after {iterations} iterations.\n"
        
    except Exception as e:
        output_string += f"\n--- AN ERROR OCCURRED IN USE CASE ---\n{str(e)}\n"

    return output_string, results_dict