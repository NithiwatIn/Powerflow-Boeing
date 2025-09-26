# microgrid_project/simulation/ybus_builder.py

import numpy as np
import pandas as pd

def build_ybus(bus_data: pd.DataFrame, line_data: pd.DataFrame) -> np.ndarray:
    """
    สร้าง Nodal Admittance Matrix (Y-bus)
    **เวอร์ชันนี้อ่านค่า Shunt G, B (p.u.) โดยตรงจาก bus_data**
    """
    num_buses = bus_data['BusID'].max()
    y_bus = np.zeros((num_buses, num_buses), dtype=complex)

    # --- ส่วนที่ 1: คำนวณจากข้อมูล Branch (Line/Transformer) ---
    # Loop ผ่านแต่ละสายส่ง
    for index, branch in line_data.iterrows():
        # ... (โค้ดส่วนนี้เหมือนเดิมทุกประการ ไม่ต้องแก้ไข) ...
        i = int(branch['FromBus']) - 1
        k = int(branch['ToBus']) - 1
        z_series = complex(branch['R_pu'], branch['X_pu'])
        if z_series == 0: 
            continue
        y_series = 1 / z_series
        tap_ratio = branch.get('TapRatio', 1.0)
        if pd.isna(tap_ratio):
            tap_ratio = 1.0
        b_pu = branch.get('B_pu', 0.0)
        y_shunt = complex(0, b_pu)
        y_bus[i, k] -= y_series / tap_ratio
        y_bus[k, i] -= y_series / tap_ratio
        y_bus[i, i] += (y_series / (tap_ratio**2)) + y_shunt
        y_bus[k, k] += y_series + y_shunt

    # --- ส่วนที่ 2: เพิ่ม Shunt Admittance จากข้อมูลบัส (Bus Data) ---
    for index, bus in bus_data.iterrows():
        bus_idx = int(bus['BusID']) - 1
        
        # ดึงค่า shunt G, B (p.u.) โดยตรง (ถ้าไม่มีให้เป็น 0)
        g_shunt_pu = bus.get('G_shunt_pu', 0.0)
        b_shunt_pu = bus.get('B_shunt_pu', 0.0)
        
        if g_shunt_pu != 0.0 or b_shunt_pu != 0.0:
            y_bus[bus_idx, bus_idx] += complex(g_shunt_pu, b_shunt_pu)

    return y_bus