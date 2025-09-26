# microgrid_project/main.py

import numpy as np
import pandas as pd
from utils.data_manager import load_microgrid_data
from simulation.ybus_builder import build_ybus
from simulation.load_flow import run_newton_raphson

def run_simulation():
    """
    ฟังก์ชันหลักสำหรับควบคุมการจำลอง
    """
    DATA_PATH = 'data/ieee-30/'
    BASE_MVA = 100.0

    try:
        # 1. อ่านข้อมูลระบบ
        system_data = load_microgrid_data(DATA_PATH)
        print("\n[1] การอ่านข้อมูลสำเร็จเรียบร้อย!")

        # 2. สร้าง Y-bus Matrix
        ybus_matrix = build_ybus(
            bus_data=system_data['buses'], 
            line_data=system_data['lines']
        )
        print("[2] สร้าง Y-bus Matrix สำเร็จ!")
        
        # 3. รัน Newton-Raphson Load Flow
        print("\n[3] กำลังคำนวณ Load Flow (Distributed Slack Bus)...")
        converged, results, iterations, losses = run_newton_raphson(
            bus_data=system_data['buses'],
            gen_data=system_data['generators'],
            load_data=system_data['loads'],
            y_bus=ybus_matrix,
            base_mva=BASE_MVA
        )

        # 4. แสดงผลลัพธ์
        print("\n" + "="*80)
        print(" " * 25 + "สรุปผลการคำนวณ Load Flow")
        print("="*80)

        if converged:
            print(f"สถานะ: สำเร็จ (Converged in {iterations} iterations)")
            print(f"กำลังสูญเสียในระบบ (P_loss): {losses:.4f} MW")
            
            pd.set_option('display.precision', 4)
            pd.set_option('display.width', 120)

            print("\nผลลัพธ์สุดท้ายที่แต่ละบัส:")
            display_cols = [
                'BusID', 'Type', 'V_final_pu', 'Angle_final_deg', 
                'Pg_final_MW', 'Qg_final_MVAR', 'Pd_final_MW', 'Qd_final_MVAR'
            ]
            print(results[display_cols])
        else:
            print(f"สถานะ: ไม่สำเร็จ (Did not converge)")
            print(f"หยุดการทำงานหลังจากคำนวณไป {iterations} รอบ")
        
        print("="*80)

        #print(ybus_matrix)

    except FileNotFoundError as e:
        print(f"\nเกิดข้อผิดพลาด: {e}")
    except Exception as e:
        print(f"\nเกิดข้อผิดพลาดที่ไม่คาดคิด: {e}")

if __name__ == "__main__":
    run_simulation()