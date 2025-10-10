# microgrid_project/utils/data_manager.py

import pandas as pd
import os

def find_available_models(data_folder_path: str) -> list:
    """
    ค้นหาโฟลเดอร์โมเดลทั้งหมดที่มีอยู่ใน data directory
    """
    try:
        models = [d for d in os.listdir(data_folder_path) 
                  if os.path.isdir(os.path.join(data_folder_path, d))]
        # กรองไฟล์ที่ไม่ใช่โมเดลออก (เช่น __pycache__)
        models = [m for m in models if not m.startswith('__')]
        return models if models else ["No models found"]
    except FileNotFoundError:
        return ["Data folder not found"]

def load_microgrid_data(model_folder_path: str) -> dict:
    """
    อ่านข้อมูลไมโครกริดทั้งหมดจากโฟลเดอร์ของโมเดลที่ระบุ
    และอ่านไฟล์อื่นๆ จาก root data folder
    """
    required_files_in_model = {
        'buses': 'bus_data.csv',
        'lines': 'line_data.csv',
        'generators': 'generator_data.csv',
        'loads': 'load_data.csv'
    }

    microgrid_data = {}
    print(f"กำลังอ่านข้อมูลจากโฟลเดอร์โมเดล: {model_folder_path}")

    # 1. อ่านไฟล์ที่จำเป็นจากโฟลเดอร์ของโมเดล (เช่น data/ieee-30/)
    for name, filename in required_files_in_model.items():
        file_path = os.path.join(model_folder_path, filename)
        try:
            df = pd.read_csv(file_path)
            microgrid_data[name] = df
            print(f"  - อ่านไฟล์ '{filename}' สำเร็จ")
        except FileNotFoundError:
            print(f"  - [ERROR] ไม่พบไฟล์ที่จำเป็น: {file_path}")
            raise FileNotFoundError(f"ไม่สามารถหาไฟล์ข้อมูลที่จำเป็น: {file_path}")
    
    # --- จุดที่แก้ไข: เปลี่ยนตำแหน่งการค้นหา system_config.csv ---
    # 2. อ่านไฟล์ system_config.csv จาก "ภายใน" โฟลเดอร์ของโมเดล
    config_path = os.path.join(model_folder_path, 'system_config.csv')
    config_dict = {}
    if os.path.exists(config_path):
        try:
            df_config = pd.read_csv(config_path)
            config_dict = pd.Series(df_config.Value.values, index=df_config.Parameter).to_dict()
            print(f"  - อ่านไฟล์ 'system_config.csv' สำเร็จ")
        except Exception as e:
            print(f"  - [Error] ไม่สามารถอ่าน 'system_config.csv': {e}")
    else:
        print(f"  - [Warning] ไม่พบไฟล์ 'system_config.csv' ใน {model_folder_path}")

    microgrid_data['config'] = config_dict
    
    # 3. อ่านไฟล์ load profile จากโฟลเดอร์ data/ หลัก
    root_data_folder = os.path.dirname(model_folder_path)
    profile_path = os.path.join(root_data_folder, 'load_profile_pattern.csv')
    
    if os.path.exists(profile_path):
        microgrid_data['load_profile'] = pd.read_csv(profile_path)
        print(f"  - อ่านไฟล์ 'load_profile_pattern.csv' สำเร็จ")
    else:
        print(f"  - [Warning] ไม่พบไฟล์ 'load_profile_pattern.csv'")
        microgrid_data['load_profile'] = None

    return microgrid_data