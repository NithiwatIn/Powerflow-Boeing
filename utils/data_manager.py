# microgrid_project/utils/data_manager.py

import pandas as pd
import os

def load_microgrid_data(data_folder_path: str) -> dict:
    """
    อ่านข้อมูลไมโครกริดทั้งหมดจากไฟล์ CSV ในโฟลเดอร์ที่ระบุ

    Args:
        data_folder_path (str): เส้นทางไปยังโฟลเดอร์ที่เก็บไฟล์ CSV

    Returns:
        dict: Dictionary ที่มี key เป็นชื่อของข้อมูล (เช่น 'buses', 'lines')
              และ value เป็น Pandas DataFrame ของข้อมูลนั้นๆ
              หากไฟล์ใดไฟล์หนึ่งไม่พบ จะเกิด Exception
    """
    # กำหนดชื่อไฟล์ที่ต้องการอ่านและ key ที่จะใช้ใน dictionary
    files_to_load = {
        'buses': 'bus_data.csv',
        'lines': 'line_data.csv',
        'generators': 'generator_data.csv',
        'loads': 'load_data.csv'
    }

    microgrid_data = {}
    print(f"กำลังอ่านข้อมูลจากโฟลเดอร์: {data_folder_path}")

    for name, filename in files_to_load.items():
        file_path = os.path.join(data_folder_path, filename)
        try:
            # ใช้ Pandas อ่านไฟล์ CSV
            df = pd.read_csv(file_path)
            microgrid_data[name] = df
            print(f"  - อ่านไฟล์ '{filename}' สำเร็จ (พบ {len(df)} แถว)")
        except FileNotFoundError:
            print(f"  - [Error] ไม่พบไฟล์: {file_path}")
            # ในโปรแกรมจริงอาจจะ raise exception เพื่อหยุดการทำงาน
            raise FileNotFoundError(f"ไม่สามารถหาไฟล์ข้อมูลที่จำเป็น: {file_path}")

    return microgrid_data