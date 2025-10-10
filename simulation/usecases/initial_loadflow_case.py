# microgrid_project/simulation/use_cases/initial_load_flow_case.py

import pandas as pd
from tabulate import tabulate

from ..ybus_builder import build_ybus
from ..newtonrapson_loadflow import run_newton_raphson

def run(system_data: dict) -> tuple:
    """
    รัน Use Case: Initial Load Flow
    - คืนค่า (string_summary, result_dataframe)
    """
    output_string = ""
    final_results_df = None  # เริ่มต้นเป็น None
    try:
        output_string += "[1] Building Y-Bus Matrix...\n"
        ybus_matrix = build_ybus(system_data['buses'], system_data['lines'])
        output_string += "    Y-Bus built successfully.\n\n"

        output_string += "[2] Running Newton-Raphson Load Flow...\n"
        BASE_MVA = 100.0
        
        converged, results_df, iterations, losses = run_newton_raphson(
            bus_data=system_data['buses'],
            gen_data=system_data['generators'],
            load_data=system_data['loads'],
            y_bus=ybus_matrix,
            base_mva=BASE_MVA
        )
        
        output_string += "-" * 80 + "\n"
        if converged:
            final_results_df = results_df # เก็บ DataFrame ผลลัพธ์เมื่อสำเร็จ
            summary = (
                f"Status: Converged in {iterations} iterations.\n"
                f"System Losses (P_loss): {losses:.4f} MW\n"
            )
            output_string += summary
            output_string += "\n--- Final Bus Results Summary ---\n"
            
            # (ส่วนของการจัดรูปแบบตารางด้วย tabulate เหมือนเดิม)
            display_df = results_df[[
                'BusID', 'Type', 'V_final_pu', 'Angle_final_deg', 
                'Pg_final_MW', 'Qg_final_MVAR', 'Pd_final_MW', 'Qd_final_MVAR'
            ]].copy()
            # ... (formatters and tabulate call)
            table_string = tabulate(display_df, headers='keys', tablefmt='psql', showindex=False, floatfmt=".4f")
            output_string += table_string + "\n"
            
        else:
            output_string += "Status: Did not converge.\n"
        
        output_string += "-" * 80 + "\n"
        
    except Exception as e:
        output_string += f"\n--- AN ERROR OCCURRED IN USE CASE ---\n{str(e)}\n"

    # คืนค่าทั้งสตริงสรุปผล และ DataFrame ผลลัพธ์ (ซึ่งอาจเป็น None ถ้าไม่สำเร็จ)
    return output_string, final_results_df