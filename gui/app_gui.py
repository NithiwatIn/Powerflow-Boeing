# microgrid_project/gui/app_gui.py

import customtkinter as ctk
import threading
import sys, os
from datetime import datetime
import pandas as pd
from tabulate import tabulate

# เพิ่ม path ของ root directory เข้าไปใน sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from utils.data_manager import find_available_models
from simulation.controller import SimulationController

# Matplotlib imports
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates

FONT_CANDIDATES = ("Courier New", "Consolas", "Menlo", "monospace")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Microgrid Time-Series Simulation")
        self.geometry("1200x800")
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.last_results_data = None
        self.DATA_PATH = os.path.join(root_dir, 'data')
        self.RESULTS_PATH = os.path.join(root_dir, 'results')
        self.controller = SimulationController(self.DATA_PATH, self.RESULTS_PATH)
        self.mono_font = ctk.CTkFont(family=FONT_CANDIDATES[0], size=11)
        
        # --- จุดที่แก้ไข: เราจะเรียกใช้ create_widgets() ที่นี่ ---
        self.create_widgets()

    def create_widgets(self):
        # --- Top Control Frame ---
        control_frame = ctk.CTkFrame(self)
        control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        control_frame.grid_columnconfigure(1, weight=1)

        # Model Selection
        ctk.CTkLabel(control_frame, text="Select Model:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.model_options = find_available_models(self.DATA_PATH)
        self.model_var = ctk.StringVar(value=self.model_options[0])
        self.model_menu = ctk.CTkOptionMenu(control_frame, variable=self.model_var, values=self.model_options)
        self.model_menu.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        
        # Use Case Selection
        ctk.CTkLabel(control_frame, text="Select Use Case:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.use_case_options = list(self.controller.use_case_map.keys())
        self.use_case_var = ctk.StringVar(value=self.use_case_options[0])
        self.use_case_menu = ctk.CTkOptionMenu(control_frame, variable=self.use_case_var, values=self.use_case_options)
        self.use_case_menu.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

        # --- จุดที่แก้ไข: สร้างปุ่มโดยไม่กำหนด command ก่อน ---
        # Run Button
        self.run_button = ctk.CTkButton(control_frame, text="Run Simulation")
        self.run_button.grid(row=0, column=2, rowspan=2, padx=10, pady=10, sticky="ns")

        # Save Frame
        save_frame = ctk.CTkFrame(control_frame)
        save_frame.grid(row=0, column=3, rowspan=2, padx=10, pady=10, sticky="ew")
        save_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(save_frame, text="Save Filename:").grid(row=0, column=0, padx=(10, 5), pady=10)
        self.filename_entry = ctk.CTkEntry(save_frame, placeholder_text="e.g., ieee30_results.csv")
        self.filename_entry.grid(row=0, column=1, padx=5, pady=10, sticky="ew")
        self.save_button = ctk.CTkButton(save_frame, text="Save Results", state="disabled")
        self.save_button.grid(row=0, column=2, padx=(5, 10), pady=10)

        # --- Main Content Frame ---
        self.main_content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_content_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        self.main_content_frame.grid_rowconfigure(0, weight=1)
        self.main_content_frame.grid_columnconfigure(0, weight=1)

        # Initial placeholder
        self.output_placeholder = ctk.CTkLabel(self.main_content_frame, text="Run a simulation to see results here.", font=("Arial", 18))
        self.output_placeholder.grid(row=0, column=0, sticky="nsew")

        # --- จุดที่แก้ไข: กำหนด command หลังจากสร้าง widget ทั้งหมดแล้ว ---
        self.run_button.configure(command=self.run_simulation_thread)
        self.save_button.configure(command=self.save_results_to_file)

    # ... (เมธอดที่เหลือทั้งหมดในคลาส App เหมือนเดิมทุกประการ ไม่ต้องแก้ไข) ...
    def clear_main_content(self):
        for widget in self.main_content_frame.winfo_children():
            widget.destroy()

    def display_text_results(self, text_content):
        self.clear_main_content()
        textbox = ctk.CTkTextbox(self.main_content_frame, wrap="none", font=self.mono_font)
        textbox.grid(row=0, column=0, sticky="nsew")
        textbox.insert("1.0", text_content)
        textbox.configure(state="disabled")
        
    def display_continuous_flow_results(self, results_dict):
        self.clear_main_content()
        tab_view = ctk.CTkTabview(self.main_content_frame)
        tab_view.grid(row=0, column=0, sticky="nsew")
        tab_view.add("Graphs")
        tab_view.add("Bus Data Tables")
        self.create_graphs(tab_view.tab("Graphs"), results_dict['plot_data'])
        self.create_tables(tab_view.tab("Bus Data Tables"), results_dict)

    def create_graphs(self, parent_frame, plot_data):
        parent_frame.grid_columnconfigure(0, weight=1)
        parent_frame.grid_rowconfigure(0, weight=1)
        fig = Figure(figsize=(10, 7), dpi=100, facecolor='#2b2b2b')
        xfmt = mdates.DateFormatter('%H:%M')
        
        ax1 = fig.add_subplot(311)
        plot_data['voltages'].plot(ax=ax1, legend=False, color='cyan')
        ax1.set_title("Bus Voltage (p.u.) vs. Time", color='white')
        ax1.set_ylabel("Voltage (p.u.)", color='white')
        ax1.xaxis.set_major_formatter(xfmt)
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax1.tick_params(axis='x', colors='white')
        ax1.tick_params(axis='y', colors='white')
        ax1.set_facecolor('#343638')

        ax2 = fig.add_subplot(312, sharex=ax1)
        plot_data['angles'].plot(ax=ax2, legend=False, color='lime')
        ax2.set_title("Bus Angle (degrees) vs. Time", color='white')
        ax2.set_ylabel("Angle (°)", color='white')
        ax2.xaxis.set_major_formatter(xfmt)
        ax2.grid(True, linestyle='--', alpha=0.3)
        ax2.tick_params(axis='x', colors='white')
        ax2.tick_params(axis='y', colors='white')
        ax2.set_facecolor('#343638')
        
        ax3 = fig.add_subplot(313, sharex=ax1)
        plot_data['total_load_mw'].plot(ax=ax3, legend=False, color='magenta')
        ax3.set_title("Total System Load (MW) vs. Time", color='white')
        ax3.set_ylabel("Load (MW)", color='white')
        ax3.set_xlabel("Time (HH:MM)", color='white')
        ax3.xaxis.set_major_formatter(xfmt)
        ax3.grid(True, linestyle='--', alpha=0.3)
        ax3.tick_params(axis='x', colors='white')
        ax3.tick_params(axis='y', colors='white')
        ax3.set_facecolor('#343638')

        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def create_tables(self, parent_frame, results_dict):
        parent_frame.grid_columnconfigure(0, weight=1)
        parent_frame.grid_rowconfigure(0, weight=1)
        scrollable_frame = ctk.CTkScrollableFrame(parent_frame, label_text="Per-Bus Time Series Data (showing first 20 time steps)")
        scrollable_frame.grid(row=0, column=0, sticky="nsew")

        grouped = results_dict['full_df'].groupby('BusID')
        cols = 3

        for i, bus_id in enumerate(sorted(grouped.groups.keys())):
            bus_df = grouped.get_group(bus_id)[['time', 'V_final_pu', 'Angle_final_deg', 'Pd_final_MW']]
            row, col = divmod(i, cols)
            bus_frame = ctk.CTkFrame(scrollable_frame, border_width=1)
            bus_frame.grid(row=row, column=col, padx=5, pady=5, sticky="n")

            ctk.CTkLabel(bus_frame, text=f"Bus {bus_id}", font=("Arial", 14, "bold")).pack(pady=(5,0))
            
            table_str = tabulate(
                bus_df.head(20),
                headers='keys', 
                tablefmt='psql', 
                showindex=False, 
                floatfmt=".4f"
            )
            
            table_label = ctk.CTkLabel(bus_frame, text=table_str, font=self.mono_font, justify="left")
            table_label.pack(padx=5, pady=5, fill="x", expand=True)

    def run_simulation_thread(self):
        self.run_button.configure(state="disabled", text="Running...")
        thread = threading.Thread(target=self.run_simulation_logic)
        thread.start()

    def run_simulation_logic(self):
        self.save_button.configure(state="disabled")
        self.last_results_data = None
        self.clear_main_content()
        loading_label = ctk.CTkLabel(self.main_content_frame, text="Simulating, please wait...", font=("Arial", 18))
        loading_label.grid(row=0, column=0, sticky="nsew")
        self.update_idletasks()

        selected_model = self.model_var.get()
        selected_use_case = self.use_case_var.get()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suggested_filename = f"{selected_model}_{selected_use_case.replace(' ', '')}_{timestamp}.csv"
        self.filename_entry.delete(0, "end")
        self.filename_entry.insert(0, suggested_filename)
        
        final_output_string, results_data = self.controller.run_use_case(selected_model, selected_use_case)
        
        if results_data is not None:
            self.last_results_data = results_data
            self.save_button.configure(state="normal")
            
            if selected_use_case == "Continuous Load Flow":
                self.display_continuous_flow_results(results_data)
            else:
                self.display_text_results(final_output_string)
        else:
             self.display_text_results(final_output_string)
        
        self.run_button.configure(state="normal", text="Run Simulation")

    def save_results_to_file(self):
        if self.last_results_data is None:
            return

        filename = self.filename_entry.get()
        if not filename:
            return

        if not filename.lower().endswith('.csv'):
            filename += '.csv'

        filepath = os.path.join(self.RESULTS_PATH, filename)
        
        try:
            if isinstance(self.last_results_data, dict) and 'full_df' in self.last_results_data:
                self.last_results_data['full_df'].to_csv(filepath, index=False)
            elif isinstance(self.last_results_data, pd.DataFrame):
                self.last_results_data.to_csv(filepath, index=False)
            
            self.save_button.configure(text="Saved!", fg_color="green")
            self.after(2000, lambda: self.save_button.configure(text="Save Results", fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"]))

        except Exception as e:
            self.save_button.configure(text="Error!", fg_color="red")
            self.after(2000, lambda: self.save_button.configure(text="Save Results", fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"]))

if __name__ == "__main__":
    app = App()
    app.mainloop()