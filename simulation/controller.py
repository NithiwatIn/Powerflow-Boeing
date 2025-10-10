# microgrid_project/simulation/controller.py

import os
import sys
from utils.data_manager import load_microgrid_data
from simulation.usecases import initial_loadflow_case
from simulation.usecases import continuous_loadflow_case 
from simulation.usecases import mpg_disconnection_case 

class SimulationController:
    def __init__(self, data_path: str, results_path: str):
        self.data_path = data_path
        self.results_path = results_path
        os.makedirs(self.results_path, exist_ok=True)
        self.use_case_map = {
            "Initial Load Flow": initial_loadflow_case.run,
            "Continuous Load Flow": continuous_loadflow_case.run,
            "MPG Disconnection": mpg_disconnection_case.run,
        }

    def run_use_case(self, model_name: str, use_case_name: str) -> tuple:
        """
        ฟังก์ชันหลักที่ GUI จะเรียกใช้
        คืนค่าเป็น tuple: (output_string, result_dataframe)
        """
        output = f"Controller: Preparing to run '{use_case_name}' on model '{model_name}'...\n"
        output += "="*60 + "\n"
        result_df = None
        
        try:
            model_folder_path = os.path.join(self.data_path, model_name)
            system_data = load_microgrid_data(model_folder_path)

            if use_case_name in self.use_case_map:
                selected_function = self.use_case_map[use_case_name]
                # รับค่า tuple กลับมา
                result_string, result_df = selected_function(system_data)
                output += result_string
            else:
                output += f"ERROR: Use case '{use_case_name}' is not defined in the controller.\n"
        
        except Exception as e:
            output += f"--- A CRITICAL ERROR OCCURRED IN CONTROLLER ---\n{str(e)}\n"
            
        return output, result_df