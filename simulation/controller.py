# simulation/controller.py

import os
from utils.data_manager import load_microgrid_data
from simulation.usecases import initial_loadflow_case
from simulation.usecases import continuous_loadflow_case
from simulation.usecases import iterative_dispatch_case 
from simulation.usecases import load_shedding_normal_case
from simulation.usecases import load_shedding_percentage_case
from simulation.usecases import load_shedding_adaptive_case

class SimulationController:
    def __init__(self, data_path: str, results_path: str):
        self.data_path = data_path
        self.results_path = results_path
        os.makedirs(self.results_path, exist_ok=True)
        
        # --- ส่วนที่แก้ไข: เพิ่ม use case ใหม่เข้าไปใน map ---
        self.use_case_map = {
            "Initial Load Flow": initial_loadflow_case.run,
            "Continuous Load Flow": continuous_loadflow_case.run,
            "MPG Disconnection (Iterative Dispatch)": iterative_dispatch_case.run,
            "Load Shedding (Normal)": load_shedding_normal_case.run,
            "Load Shedding (Percentage)": load_shedding_percentage_case.run,
            "Load Shedding (Adaptive)": load_shedding_adaptive_case.run,
        }

    def run_use_case(self, use_case_name: str, system_data: dict) -> tuple:
        output = f"Controller: Preparing to run '{use_case_name}'...\n"
        output += "="*60 + "\n"
        result_data = None
        
        try:
            if use_case_name in self.use_case_map:
                selected_function = self.use_case_map[use_case_name]
                # ส่ง system_data เข้าไปใน use case เลย
                output_str, result_data = selected_function(system_data)
                output += output_str
            else:
                output += f"ERROR: Use case '{use_case_name}' is not defined in the controller.\n"
        
        except Exception as e:
            output += f"--- A CRITICAL ERROR OCCURRED IN CONTROLLER ---\n{str(e)}\n"
            import traceback
            output += traceback.format_exc()

        return output, result_data