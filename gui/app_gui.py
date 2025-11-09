# microgrid_project/gui/app_gui.py

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import customtkinter as ctk
import pandas as pd
import os
import threading
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tabulate import tabulate

from simulation.controller import SimulationController
from utils.data_manager import find_available_models, load_microgrid_data

modern_luxury_style = {
    "figure.facecolor": "#1D2025", "axes.facecolor": "#21252B", "axes.edgecolor": "#ABB2BF", 
    "axes.labelcolor": "#ABB2BF", "axes.titlecolor": "white", "xtick.color": "#ABB2BF", 
    "ytick.color": "#ABB2BF", "grid.color": "#4B5263", "grid.linestyle": "--", 
    "grid.alpha": 0.3, "text.color": "#ABB2BF",
}
plt.rcParams.update(modern_luxury_style)

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark"); ctk.set_default_color_theme("blue")
        self.title("Microgrid Simulation Dashboard")
        self.geometry("1400x900")
        self.grid_columnconfigure(0, weight=1); self.grid_rowconfigure(1, weight=1)
        self.DATA_PATH = 'data'; self.RESULTS_PATH = 'results'
        os.makedirs(self.RESULTS_PATH, exist_ok=True)
        self.controller = SimulationController(self.DATA_PATH, self.RESULTS_PATH)
        self.current_system_data = None; self.last_results_data = None
        self.plotted_lines = {}; self.summary_labels = {}
        self.hover_line = None; self.hover_points = []
        self.annotation = None; self.selected_line = None; self.pinned_annotation = None
        self.iter_hover_line_power = None; self.iter_hover_line_freq = None
        self.iter_hover_point = None; self.iter_annotation = None
        self.ls_hover_line_power = None; self.ls_hover_line_freq = None
        self.ls_hover_point = None; self.ls_annotation = None
        
        self.create_widgets()

    def create_widgets(self):
        control_frame = ctk.CTkFrame(self)
        control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        control_frame.grid_columnconfigure(3, weight=1)
        ctk.CTkLabel(control_frame, text="Select Model:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        model_options = find_available_models(self.DATA_PATH)
        self.model_var = ctk.StringVar(value=model_options[0] if model_options else "")
        self.model_menu = ctk.CTkOptionMenu(control_frame, variable=self.model_var, values=model_options)
        self.model_menu.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        ctk.CTkLabel(control_frame, text="Select Use Case:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        use_case_options = list(self.controller.use_case_map.keys())
        self.use_case_var = ctk.StringVar(value=use_case_options[0] if use_case_options else "")
        self.use_case_menu = ctk.CTkOptionMenu(control_frame, variable=self.use_case_var, values=use_case_options)
        self.use_case_menu.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        self.run_button = ctk.CTkButton(control_frame, text="Run Simulation", command=self.run_simulation_thread)
        self.run_button.grid(row=0, column=2, rowspan=2, padx=(20, 10), pady=10, sticky="ns")
        save_frame = ctk.CTkFrame(control_frame)
        save_frame.grid(row=0, column=3, rowspan=2, padx=10, pady=10, sticky="ew")
        save_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(save_frame, text="Save Filename:").grid(row=0, column=0, padx=(10, 5), pady=10)
        self.filename_entry = ctk.CTkEntry(save_frame, placeholder_text="e.g., ieee30_results.csv")
        self.filename_entry.grid(row=0, column=1, padx=5, pady=10, sticky="ew")
        self.save_button = ctk.CTkButton(save_frame, text="Save Results", state="disabled", command=self.save_results_to_file)
        self.save_button.grid(row=0, column=2, padx=(5, 10), pady=10)
        
        self.main_content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_content_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        self.main_content_frame.grid_rowconfigure(0, weight=1); self.main_content_frame.grid_columnconfigure(0, weight=1)
        self.output_placeholder = ctk.CTkLabel(self.main_content_frame, text="Run a simulation to see results here.", font=("Arial", 18))
        self.log_textbox = ctk.CTkTextbox(self.main_content_frame, font=("Courier New", 12), wrap="none")
        self.create_continuous_view() 
        self.create_iterative_dispatch_view()
        self.create_loadshedding_view()
        self.create_percentage_shedding_view()
        self.create_adaptive_shedding_view()
        self.show_content_view("placeholder")

    def create_continuous_view(self):
        self.continuous_frame = ctk.CTkFrame(self.main_content_frame, fg_color="transparent")
        self.continuous_frame.grid_columnconfigure(0, weight=2, minsize=240); self.continuous_frame.grid_columnconfigure(1, weight=8); self.continuous_frame.grid_rowconfigure(0, weight=1)
        self.summary_panel = ctk.CTkScrollableFrame(self.continuous_frame, fg_color="#21252B", label_text="CURRENT DATA & DISPLAY", label_text_color="#E5C07B")
        self.summary_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        main_content_cont = ctk.CTkFrame(self.continuous_frame, fg_color="transparent")
        main_content_cont.grid(row=0, column=1, sticky="nsew")
        main_content_cont.grid_rowconfigure(0, weight=6); main_content_cont.grid_rowconfigure(2, weight=4); main_content_cont.grid_columnconfigure(0, weight=1)
        graph_container_cont = ctk.CTkFrame(main_content_cont, fg_color="#21252B"); graph_container_cont.grid(row=0, column=0, sticky="nsew", pady=(0,10))
        self.cont_fig = Figure(dpi=100); self.cont_ax = self.cont_fig.add_subplot(111)
        self.cont_canvas = FigureCanvasTkAgg(self.cont_fig, master=graph_container_cont); self.cont_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.cont_table_title_var = ctk.StringVar(value="ตารางแสดงผลการรัน loadflow ณ เวลา --:--")
        ctk.CTkLabel(main_content_cont, textvariable=self.cont_table_title_var, font=("Arial", 14)).grid(row=1, column=0, sticky="w")
        table_container_cont = ctk.CTkFrame(main_content_cont, fg_color="#21252B"); table_container_cont.grid(row=2, column=0, sticky="nsew", pady=(5,0))
        table_container_cont.grid_rowconfigure(0, weight=1); table_container_cont.grid_columnconfigure(0, weight=1)
        style = ttk.Style()
        style.theme_use("default"); style.configure("Treeview", background="#21252B", foreground="white", fieldbackground="#21252B", borderwidth=0, rowheight=22)
        style.map('Treeview', background=[('selected', '#003366')]); style.configure("Treeview.Heading", background="#565B5E", foreground="white", relief="flat", font=('Calibri', 10, 'bold'))
        cols_cont = ('BusID','Type','V_final_pu','Angle_final_deg','Pg_final_MW','Qg_final_MVAR','Pd_final_MW','Qd_final_MVAR')
        self.cont_tree = ttk.Treeview(table_container_cont, columns=cols_cont, show='headings', style="Treeview", height=15)
        for col in cols_cont: self.cont_tree.heading(col, text=col, anchor='center')
        self.cont_tree.column('BusID', anchor='center', width=60); self.cont_tree.column('Type', anchor='center', width=60)
        self.cont_tree.column('V_final_pu', anchor='e', width=100); self.cont_tree.column('Angle_final_deg', anchor='e', width=110)
        self.cont_tree.column('Pg_final_MW', anchor='e', width=100); self.cont_tree.column('Qg_final_MVAR', anchor='e', width=110)
        self.cont_tree.column('Pd_final_MW', anchor='e', width=100); self.cont_tree.column('Qd_final_MVAR', anchor='e', width=110)
        self.cont_tree.pack(side="top", fill="both", expand=True, padx=10, pady=10)

    def create_iterative_dispatch_view(self):
        self.iterative_frame = ctk.CTkScrollableFrame(self.main_content_frame, fg_color="transparent")
        self.iterative_frame.grid_columnconfigure(0, weight=1)
        graph_container_iter = ctk.CTkFrame(self.iterative_frame, fg_color="#21252B")
        graph_container_iter.grid(row=0, column=0, sticky="ew", padx=10, pady=10); graph_container_iter.grid_columnconfigure(0, weight=1)
        self.iter_fig = Figure(figsize=(12, 8), dpi=100)
        self.iter_ax_power = self.iter_fig.add_subplot(211); self.iter_ax_freq = self.iter_fig.add_subplot(212)
        self.iter_canvas = FigureCanvasTkAgg(self.iter_fig, master=graph_container_iter); self.iter_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.explanation_textbox = ctk.CTkTextbox(self.iterative_frame, font=("Courier New", 11), wrap="word", height=500)
        self.explanation_textbox.grid(row=1, column=0, sticky="ew", padx=10, pady=10); self.explanation_textbox.configure(state="disabled")
        self.iter_table_title_var = ctk.StringVar(value="ตารางแสดงผลการรัน loadflow ณ เวลา --:--")
        ctk.CTkLabel(self.iterative_frame, textvariable=self.iter_table_title_var, font=("Arial", 14)).grid(row=2, column=0, sticky="w", padx=10, pady=(10,0))
        table_container_iter = ctk.CTkFrame(self.iterative_frame, fg_color="#21252B")
        table_container_iter.grid(row=3, column=0, sticky="ew", padx=10, pady=5)
        table_container_iter.grid_rowconfigure(0, weight=1); table_container_iter.grid_columnconfigure(0, weight=1)
        cols_iter = ('BusID','Type','V_final_pu','Angle_final_deg','Pg_final_MW','Qg_final_MVAR','Pd_final_MW','Qd_final_MVAR','Frequency_Hz')
        self.iter_tree = ttk.Treeview(table_container_iter, columns=cols_iter, show='headings', style="Treeview", height=15)
        for col in cols_iter: self.iter_tree.heading(col, text=col, anchor='center')
        self.iter_tree.pack(side="top", fill="both", expand=True, padx=10, pady=10)

    def create_loadshedding_view(self):
        self.loadshedding_frame = ctk.CTkScrollableFrame(self.main_content_frame, fg_color="transparent")
        self.loadshedding_frame.grid_columnconfigure(0, weight=1)
        graph_container_ls = ctk.CTkFrame(self.loadshedding_frame, fg_color="#21252B")
        graph_container_ls.grid(row=0, column=0, sticky="ew", padx=10, pady=10); graph_container_ls.grid_columnconfigure(0, weight=1)
        self.ls_fig = Figure(figsize=(12, 8), dpi=100)
        self.ls_ax_power = self.ls_fig.add_subplot(211); self.ls_ax_freq = self.ls_fig.add_subplot(212)
        self.ls_canvas = FigureCanvasTkAgg(self.ls_fig, master=graph_container_ls); self.ls_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.ls_table_title_var = ctk.StringVar(value="ตารางบันทึกการตัดโหลด (Load Shedding Log) ณ เวลา --:--")
        ctk.CTkLabel(self.loadshedding_frame, textvariable=self.ls_table_title_var, font=("Arial", 14)).grid(row=2, column=0, sticky="w", padx=10, pady=(10,0))
        table_container_ls = ctk.CTkFrame(self.loadshedding_frame, fg_color="#21252B")
        table_container_ls.grid(row=3, column=0, sticky="ew", padx=10, pady=5)
        table_container_ls.grid_rowconfigure(0, weight=1); table_container_ls.grid_columnconfigure(0, weight=1)
        
        # --- ส่วนที่แก้ไข: เพิ่ม 'Priority' ---
        cols_ls = ('ลำดับ', 'BusID', 'Priority', 'MW', 'MVA')
        self.ls_tree = ttk.Treeview(table_container_ls, columns=cols_ls, show='headings', style="Treeview", height=10)
        for col in cols_ls: self.ls_tree.heading(col, text=col, anchor='center')
        self.ls_tree.column('ลำดับ', anchor='center', width=50); self.ls_tree.column('BusID', anchor='center', width=80)
        self.ls_tree.column('Priority', anchor='center', width=80); self.ls_tree.column('MW', anchor='center', width=150); self.ls_tree.column('MVA', anchor='center', width=150)
        self.ls_tree.pack(side="top", fill="both", expand=True, padx=10, pady=10)

    def create_percentage_shedding_view(self):
        # สร้าง Frame ที่เหมือนกับ load_shedding_view
        self.perc_shed_frame = ctk.CTkScrollableFrame(self.main_content_frame, fg_color="transparent")
        self.perc_shed_frame.grid_columnconfigure(0, weight=1)
        graph_container_ps = ctk.CTkFrame(self.perc_shed_frame, fg_color="#21252B")
        graph_container_ps.grid(row=0, column=0, sticky="ew", padx=10, pady=10); graph_container_ps.grid_columnconfigure(0, weight=1)
        self.ps_fig = Figure(figsize=(12, 8), dpi=100)
        self.ps_ax_power = self.ps_fig.add_subplot(211); self.ps_ax_freq = self.ps_fig.add_subplot(212)
        self.ps_canvas = FigureCanvasTkAgg(self.ps_fig, master=graph_container_ps); self.ps_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.ps_table_title_var = ctk.StringVar(value="ตารางบันทึกการตัดโหลด (Percentage) ณ เวลา --:--")
        ctk.CTkLabel(self.perc_shed_frame, textvariable=self.ps_table_title_var, font=("Arial", 14)).grid(row=2, column=0, sticky="w", padx=10, pady=(10,0))
        table_container_ps = ctk.CTkFrame(self.perc_shed_frame, fg_color="#21252B")
        table_container_ps.grid(row=3, column=0, sticky="ew", padx=10, pady=5)
        table_container_ps.grid_rowconfigure(0, weight=1); table_container_ps.grid_columnconfigure(0, weight=1)
        
        # --- ตารางใหม่พร้อมคอลัมน์ % Shed ---
        cols_ps = ('ลำดับ', 'BusID', 'Priority', '% Shed', 'MW', 'MVA')
        self.ps_tree = ttk.Treeview(table_container_ps, columns=cols_ps, show='headings', style="Treeview", height=10)
        for col in cols_ps: self.ps_tree.heading(col, text=col, anchor='center')
        self.ps_tree.column('ลำดับ', anchor='center', width=50); self.ps_tree.column('BusID', anchor='center', width=80)
        self.ps_tree.column('Priority', anchor='center', width=80); self.ps_tree.column('% Shed', anchor='e', width=100)
        self.ps_tree.column('MW', anchor='e', width=150); self.ps_tree.column('MVA', anchor='e', width=150)
        self.ps_tree.pack(side="top", fill="both", expand=True, padx=10, pady=10)

    def create_adaptive_shedding_view(self):
        # สร้าง Frame ที่เหมือนกับ percentage_shedding_view
        self.adaptive_frame = ctk.CTkScrollableFrame(self.main_content_frame, fg_color="transparent")
        self.adaptive_frame.grid_columnconfigure(0, weight=1)
        graph_container_as = ctk.CTkFrame(self.adaptive_frame, fg_color="#21252B")
        graph_container_as.grid(row=0, column=0, sticky="ew", padx=10, pady=10); graph_container_as.grid_columnconfigure(0, weight=1)
        self.as_fig = Figure(figsize=(12, 8), dpi=100)
        self.as_ax_power = self.as_fig.add_subplot(211); self.as_ax_freq = self.as_fig.add_subplot(212)
        self.as_canvas = FigureCanvasTkAgg(self.as_fig, master=graph_container_as); self.as_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.as_table_title_var = ctk.StringVar(value="ตารางบันทึกการตัดโหลด (Adaptive) ณ เวลา --:--")
        ctk.CTkLabel(self.adaptive_frame, textvariable=self.as_table_title_var, font=("Arial", 14)).grid(row=2, column=0, sticky="w", padx=10, pady=(10,0))
        table_container_as = ctk.CTkFrame(self.adaptive_frame, fg_color="#21252B")
        table_container_as.grid(row=3, column=0, sticky="ew", padx=10, pady=5)
        table_container_as.grid_rowconfigure(0, weight=1); table_container_as.grid_columnconfigure(0, weight=1)
        
        # --- ตารางใหม่พร้อมคอลัมน์ Priority แบบใหม่ ---
        cols_as = ('ลำดับ', 'BusID', 'Priority', '% Shed', 'MW', 'MVA')
        self.as_tree = ttk.Treeview(table_container_as, columns=cols_as, show='headings', style="Treeview", height=10)
        for col in cols_as: self.as_tree.heading(col, text=col, anchor='center')
        self.as_tree.column('ลำดับ', anchor='center', width=50); self.as_tree.column('BusID', anchor='center', width=80)
        self.as_tree.column('Priority', anchor='center', width=120); # <--- ขยายคอลัมน์
        self.as_tree.column('% Shed', anchor='e', width=100); self.as_tree.column('MW', anchor='e', width=150); self.as_tree.column('MVA', anchor='e', width=150)
        self.as_tree.pack(side="top", fill="both", expand=True, padx=10, pady=10)

    def show_content_view(self, view_name):
        self.output_placeholder.grid_forget(); self.log_textbox.grid_forget()
        self.continuous_frame.grid_forget(); self.iterative_frame.grid_forget(); self.loadshedding_frame.grid_forget()
        self.perc_shed_frame.grid_forget(); self.adaptive_frame.grid_forget()
        if view_name == "placeholder": self.output_placeholder.grid(row=0, column=0, sticky="nsew")
        elif view_name == "log": self.log_textbox.grid(row=0, column=0, sticky="nsew")
        elif view_name == "continuous": self.continuous_frame.grid(row=0, column=0, sticky="nsew")
        elif view_name == "iterative_dispatch": self.iterative_frame.grid(row=0, column=0, sticky="nsew")
        elif view_name == "load_shedding": self.loadshedding_frame.grid(row=0, column=0, sticky="nsew")
        elif view_name == "percentage_shedding": self.perc_shed_frame.grid(row=0, column=0, sticky="nsew")
        elif view_name == "adaptive_shedding": self.adaptive_frame.grid(row=0, column=0, sticky="nsew")

    def run_simulation_thread(self):
        thread = threading.Thread(target=self.run_simulation); thread.daemon = True; thread.start()

    def run_simulation(self):
        self.run_button.configure(text="Running...", state="disabled"); self.save_button.configure(state="disabled")
        model_name = self.model_var.get(); use_case_name = self.use_case_var.get()
        try:
            model_folder_path = os.path.join(self.DATA_PATH, model_name)
            self.current_system_data = load_microgrid_data(model_folder_path)
            output, results = self.controller.run_use_case(use_case_name, self.current_system_data)
            self.last_results_data = results
            self.after(0, self.update_gui_after_run, output, results, use_case_name)
        except Exception as e:
            self.run_button.configure(text="Run Simulation", state="normal")
            messagebox.showerror("Data Loading Error", f"Failed to load system data for '{model_name}'.\nError: {e}")

    def update_gui_after_run(self, output, results, use_case_name):
        self.run_button.configure(text="Run Simulation", state="normal")
        if results is not None:
            self.save_button.configure(state="normal")
            sanitized_uc_name = use_case_name.replace(' ', '_').replace('(', '').replace(')','')
            suggested_filename = f"{self.model_var.get()}_{sanitized_uc_name}.csv"
            self.filename_entry.delete(0, 'end'); self.filename_entry.insert(0, suggested_filename)
        if results is None:
            self.show_content_view("log"); self.log_textbox.delete("1.0", "end"); self.log_textbox.insert("1.0", output)
            return
        if use_case_name == "Initial Load Flow":
            self.show_content_view("log"); self.log_textbox.delete("1.0", "end"); self.log_textbox.insert("1.0", output)
        elif use_case_name == "Continuous Load Flow":
            self.show_content_view("continuous"); self.setup_interactive_plot(results)
        elif use_case_name == "MPG Disconnection (Iterative Dispatch)":
            self.show_content_view("iterative_dispatch"); self.setup_iterative_dispatch_view(results)
        elif use_case_name == "MPG Disconnection (Primary Freq.)":
            self.show_content_view("iterative_dispatch"); self.setup_primary_freq_view(results)
        elif use_case_name == "Load Shedding (Normal)":
            self.show_content_view("load_shedding"); self.setup_loadshedding_plot(results)
        elif use_case_name == "Load Shedding (Percentage)": 
            self.show_content_view("percentage_shedding"); self.setup_percentage_shedding_plot(results)
        elif use_case_name == "Load Shedding (Adaptive)": 
            self.show_content_view("adaptive_shedding"); self.setup_adaptive_shedding_plot(results)
        else:
            self.show_content_view("placeholder"); self.output_placeholder.configure(text=f"'{use_case_name}' result view is not implemented yet.")

    def setup_interactive_plot(self, results):
        self.interactive_plot_data = results; self.plotted_lines.clear(); self.summary_labels.clear(); self.hover_points.clear()
        self.cont_ax.clear()
        for widget in self.summary_panel.winfo_children():
            if not isinstance(widget, ctk.CTkLabel) or "CURRENT DATA" not in widget.cget("text"): widget.destroy()
        summary_data = results['summary_data']; pivoted_gens = summary_data['pivoted_gens_mw']
        self.setup_summary_and_legend_item("Total Load", "#61AFEF")
        colors = plt.cm.get_cmap('viridis', len(pivoted_gens.columns))
        for i, gen_id in enumerate(pivoted_gens.columns): self.setup_summary_and_legend_item(f'Gen {int(gen_id)}', colors(i))
        for name, checkbox_info in self.summary_labels.items():
            line_data = summary_data['total_load_mw'] if name == "Total Load" else pivoted_gens[int(name.split(' ')[1])]
            linestyle = '-' if name == "Total Load" else '--'; linewidth = 2.5 if name == "Total Load" else 1.5
            line, = self.cont_ax.plot(line_data.index, line_data, label=name, color=checkbox_info['color'], linestyle=linestyle, linewidth=linewidth, zorder=10)
            self.plotted_lines[name] = line; checkbox_info['checkbox'].configure(command=lambda n=name: self.toggle_line_visibility(n))
        self.cont_ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        self.cont_ax.set_title('Power Generation & Load Profile (24h)'); self.cont_ax.set_xlabel('Time'); self.cont_ax.set_ylabel('Power (MW)')
        self.cont_fig.tight_layout(pad=2.0)
        self.hover_line = self.cont_ax.axvline(summary_data['total_load_mw'].index[0], color='white', linestyle='dotted', linewidth=1, alpha=0.7, zorder=20, visible=False)
        for _ in self.plotted_lines:
            point, = self.cont_ax.plot([], [], 'o', markersize=8, markeredgecolor='white', zorder=30, visible=False)
            self.hover_points.append(point)
        self.annotation = self.cont_ax.annotate("", xy=(0,0), xytext=(20,-40), textcoords="offset points", bbox=dict(boxstyle="round,pad=0.5", fc="#282C34", ec="#61AFEF", lw=1, alpha=0.9), arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.1", color='white'), visible=False, zorder=40, fontname="Arial", fontsize=10)
        self.cont_canvas.mpl_connect('motion_notify_event', self.on_hover); self.cont_canvas.mpl_connect('button_press_event', self.on_click)
        last_time = summary_data['total_load_mw'].index[-1]
        self.selected_line = self.cont_ax.axvline(last_time, color='#98C379', linestyle='-', linewidth=2, zorder=25)
        self.update_summary_panel(last_time); self.update_interactive_table(last_time, self.cont_tree, self.cont_table_title_var)
        self.cont_canvas.draw()

    def setup_adaptive_shedding_plot(self, results):
        # ฟังก์ชันนี้เหมือนกับ setup_percentage_shedding_plot เกือบ 100%
        self.interactive_plot_data = results
        self.as_ax_power.clear(); self.as_ax_freq.clear()
        if hasattr(self, 'pinned_annotation_as') and self.pinned_annotation_as:
            self.pinned_annotation_as.set_visible(False)
        
        summary = results['summary_data']
        self.as_ax_power.plot(summary['total_load_mw_before'].index, summary['total_load_mw_before'], label='Total Load (Before)', color="#61AFEF", linestyle=':')
        self.as_ax_power.plot(summary['total_load_mw_after'].index, summary['total_load_mw_after'], label='Total Load (After)', color="#C678DD")
        self.as_ax_power.axvline(summary['disconnection_time'], color='red', linestyle='--', label='MPG Disconnect')
        pmax_start_time = summary['disconnection_time']; pmax_end_time = summary['total_load_mw_before'].index[-1]
        self.as_ax_power.plot([pmax_start_time, pmax_end_time], [summary['microgrid_pmax'], summary['microgrid_pmax']], color='orange', linestyle=':', label='Microgrid Pmax')
        self.as_ax_power.set_title("Power Profile (Adaptive Load Shedding)"); self.as_ax_power.set_ylabel("Power (MW)"); self.as_ax_power.legend()
        self.as_ax_power.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        self.as_ax_freq.plot(summary['frequency_series_before'].index, summary['frequency_series_before']['freq_before'], label='Frequency (Before)', color="#E06C75", linestyle=':')
        self.as_ax_freq.plot(summary['frequency_series_after'].index, summary['frequency_series_after']['freq_after'], label='Frequency (After)', color="cyan")
        self.as_ax_freq.axhline(summary['freq_threshold'], color='yellow', linestyle='--', label=f'Threshold ({summary["freq_threshold"]:.2f} Hz)')
        self.as_ax_freq.set_title("Frequency Profile (Adaptive Load Shedding)"); self.as_ax_freq.set_ylabel("Frequency (Hz)"); self.as_ax_freq.set_xlabel("Time"); self.as_ax_freq.legend()
        self.as_ax_freq.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        self.as_fig.tight_layout()
        
        last_time = summary['total_load_mw_before'].index[-1]
        self.as_selected_line_power = self.as_ax_power.axvline(last_time, color='#98C379', linestyle='-', linewidth=2, zorder=25)
        self.as_selected_line_freq = self.as_ax_freq.axvline(last_time, color='#98C379', linestyle='-', linewidth=2, zorder=25)
        
        self.pinned_annotation_as = None
        self.as_canvas.mpl_connect('button_press_event', self.on_click_adaptive_shedding)
        
        self.update_adaptive_shedding_table(last_time) # <--- เรียกฟังก์ชันใหม่
        self.as_canvas.draw()

    def setup_iterative_dispatch_view(self, results):
        self.interactive_plot_data = results
        self.iter_ax_power.clear(); self.iter_ax_freq.clear()
        if hasattr(self, 'pinned_annotation') and self.pinned_annotation: self.pinned_annotation.set_visible(False)
        summary = results['summary_data']
        self.iter_ax_power.plot(summary['total_load_mw'].index, summary['total_load_mw'], label='Total Load', color="#61AFEF")
        self.iter_ax_power.axvline(summary['disconnection_time'], color='red', linestyle='--', label='MPG Disconnect')
        pmax_start_time = summary['disconnection_time']; pmax_end_time = summary['total_load_mw'].index[-1]
        self.iter_ax_power.plot([pmax_start_time, pmax_end_time], [summary['microgrid_pmax'], summary['microgrid_pmax']], color='orange', linestyle=':', label='Microgrid Pmax')
        self.iter_ax_power.set_title("Power Profile"); self.iter_ax_power.set_ylabel("Power (MW)"); self.iter_ax_power.legend()
        self.iter_ax_power.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        self.iter_ax_freq.plot(summary['frequency_series'].index, summary['frequency_series']['frequency'], label='System Frequency', color='cyan')
        self.iter_ax_freq.set_title("Frequency Profile"); self.iter_ax_freq.set_ylabel("Frequency (Hz)"); self.iter_ax_freq.set_xlabel("Time")
        self.iter_ax_freq.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        self.iter_fig.tight_layout()
        last_time = summary['total_load_mw'].index[-1]
        self.selected_line_power = self.iter_ax_power.axvline(last_time, color='#98C379', linestyle='-', linewidth=2, zorder=25)
        self.selected_line_freq = self.iter_ax_freq.axvline(last_time, color='#98C379', linestyle='-', linewidth=2, zorder=25)
        self.pinned_annotation = None
        self.iter_hover_line_power = self.iter_ax_power.axvline(last_time, color='white', linestyle='dotted', linewidth=1, alpha=0.7, zorder=20, visible=False)
        self.iter_hover_line_freq = self.iter_ax_freq.axvline(last_time, color='white', linestyle='dotted', linewidth=1, alpha=0.7, zorder=20, visible=False)
        self.iter_hover_point, = self.iter_ax_power.plot([],[], 'o', color='#61AFEF', markersize=8, markeredgecolor='white', zorder=30, visible=False)
        self.iter_annotation = self.iter_ax_power.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                                            bbox=dict(boxstyle="round,pad=0.5", fc="#282C34", ec="#61AFEF", lw=1, alpha=0.9),
                                            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color='white'),
                                            visible=False, zorder=40, fontname="Arial", fontsize=10)
        self.iter_canvas.mpl_connect('motion_notify_event', self.on_hover_iterative)
        self.iter_canvas.mpl_connect('button_press_event', self.on_click_iterative)
        self.update_explanation_panel(last_time)
        self.update_interactive_table(last_time, self.iter_tree, self.iter_table_title_var)
        self.iter_canvas.draw()

    def setup_primary_freq_view(self, results):
        self.iter_ax_power.clear(); self.iter_ax_freq.clear()
        self.iter_ax_power.set_visible(False); self.iter_ax_freq.set_visible(True)
        self.iter_fig.subplots_adjust(top=0.9, bottom=0.15, left=0.1, right=0.95)
        freq_df = results["dynamic_freq_df"]
        self.iter_ax_freq.set_title('Primary Frequency Response after Disconnection'); self.iter_ax_freq.set_xlabel('Time (seconds)'); self.iter_ax_freq.set_ylabel('Frequency (Hz)')
        self.iter_ax_freq.plot(freq_df['time_s'], freq_df['frequency_hz'], label='System Frequency', color='#FF6347')
        self.iter_ax_freq.axhline(y=50.0, color='white', linestyle='--', label='Nominal Freq. (50 Hz)')
        nadir = freq_df['frequency_hz'].min()
        self.iter_ax_freq.axhline(y=nadir, color='yellow', linestyle=':', label=f'Nadir ({nadir:.3f} Hz)')
        self.iter_ax_freq.legend()
        self.iter_canvas.draw()

    def setup_loadshedding_plot(self, results):
        self.interactive_plot_data = results
        self.ls_ax_power.clear(); self.ls_ax_freq.clear()
        if hasattr(self, 'pinned_annotation_ls') and self.pinned_annotation_ls: self.pinned_annotation_ls.set_visible(False)
        summary = results['summary_data']
        self.ls_ax_power.plot(summary['total_load_mw_before'].index, summary['total_load_mw_before'], label='Total Load (Before)', color="#61AFEF", linestyle=':')
        self.ls_ax_power.plot(summary['total_load_mw_after'].index, summary['total_load_mw_after'], label='Total Load (After)', color="#C678DD")
        self.ls_ax_power.axvline(summary['disconnection_time'], color='red', linestyle='--', label='MPG Disconnect')
        pmax_start_time = summary['disconnection_time']; pmax_end_time = summary['total_load_mw_before'].index[-1]
        self.ls_ax_power.plot([pmax_start_time, pmax_end_time], [summary['microgrid_pmax'], summary['microgrid_pmax']], color='orange', linestyle=':', label='Microgrid Pmax')
        self.ls_ax_power.set_title("Power Profile (with Load Shedding)"); self.ls_ax_power.set_ylabel("Power (MW)"); self.ls_ax_power.legend()
        self.ls_ax_power.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        self.ls_ax_freq.plot(summary['frequency_series_before'].index, summary['frequency_series_before']['freq_before'], label='Frequency (Before)', color="#E06C75", linestyle=':')
        self.ls_ax_freq.plot(summary['frequency_series_after'].index, summary['frequency_series_after']['freq_after'], label='Frequency (After)', color="cyan")
        self.ls_ax_freq.axhline(summary['freq_threshold'], color='yellow', linestyle='--', label=f'Threshold ({summary["freq_threshold"]:.2f} Hz)')
        self.ls_ax_freq.set_title("Frequency Profile (with Load Shedding)"); self.ls_ax_freq.set_ylabel("Frequency (Hz)"); self.ls_ax_freq.set_xlabel("Time"); self.ls_ax_freq.legend()
        self.ls_ax_freq.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        self.ls_fig.tight_layout()
        last_time = summary['total_load_mw_before'].index[-1]
        self.ls_selected_line_power = self.ls_ax_power.axvline(last_time, color='#98C379', linestyle='-', linewidth=2, zorder=25)
        self.ls_selected_line_freq = self.ls_ax_freq.axvline(last_time, color='#98C379', linestyle='-', linewidth=2, zorder=25)
        
        # --- ส่วนที่แก้ไข: สร้าง Hover Elements ---
        self.ls_hover_line_power = self.ls_ax_power.axvline(last_time, color='white', linestyle='dotted', linewidth=1, alpha=0.7, zorder=20, visible=False)
        self.ls_hover_line_freq = self.ls_ax_freq.axvline(last_time, color='white', linestyle='dotted', linewidth=1, alpha=0.7, zorder=20, visible=False)
        self.ls_hover_point, = self.ls_ax_power.plot([],[], 'o', color='#C678DD', markersize=8, markeredgecolor='white', zorder=30, visible=False)
        self.ls_annotation = self.ls_ax_power.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                                            bbox=dict(boxstyle="round,pad=0.5", fc="#282C34", ec="#61AFEF", lw=1, alpha=0.9),
                                            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color='white'),
                                            visible=False, zorder=40, fontname="Arial", fontsize=10)
        self.pinned_annotation_ls = None
        
        self.ls_canvas.mpl_connect('motion_notify_event', self.on_hover_loadshedding)
        self.ls_canvas.mpl_connect('button_press_event', self.on_click_loadshedding)
        self.update_loadshedding_table(last_time)
        self.ls_canvas.draw()

    def setup_percentage_shedding_plot(self, results):
        self.interactive_plot_data = results
        self.ps_ax_power.clear(); self.ps_ax_freq.clear()
        if hasattr(self, 'pinned_annotation_ps') and self.pinned_annotation_ps:
            self.pinned_annotation_ps.set_visible(False)
        
        summary = results['summary_data']
        
        # กราฟเหมือนกับ Load Shedding (Normal)
        self.ps_ax_power.plot(summary['total_load_mw_before'].index, summary['total_load_mw_before'], label='Total Load (Before)', color="#61AFEF", linestyle=':')
        self.ps_ax_power.plot(summary['total_load_mw_after'].index, summary['total_load_mw_after'], label='Total Load (After)', color="#C678DD")
        self.ps_ax_power.axvline(summary['disconnection_time'], color='red', linestyle='--', label='MPG Disconnect')
        pmax_start_time = summary['disconnection_time']; pmax_end_time = summary['total_load_mw_before'].index[-1]
        self.ps_ax_power.plot([pmax_start_time, pmax_end_time], [summary['microgrid_pmax'], summary['microgrid_pmax']], color='orange', linestyle=':', label='Microgrid Pmax')
        self.ps_ax_power.set_title("Power Profile (Percentage Load Shedding)"); self.ps_ax_power.set_ylabel("Power (MW)"); self.ps_ax_power.legend()
        self.ps_ax_power.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        self.ps_ax_freq.plot(summary['frequency_series_before'].index, summary['frequency_series_before']['freq_before'], label='Frequency (Before)', color="#E06C75", linestyle=':')
        self.ps_ax_freq.plot(summary['frequency_series_after'].index, summary['frequency_series_after']['freq_after'], label='Frequency (After)', color="cyan")
        self.ps_ax_freq.axhline(summary['freq_threshold'], color='yellow', linestyle='--', label=f'Threshold ({summary["freq_threshold"]:.2f} Hz)')
        self.ps_ax_freq.set_title("Frequency Profile (Percentage Load Shedding)"); self.ps_ax_freq.set_ylabel("Frequency (Hz)"); self.ps_ax_freq.set_xlabel("Time"); self.ps_ax_freq.legend()
        self.ps_ax_freq.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        self.ps_fig.tight_layout()
        
        last_time = summary['total_load_mw_before'].index[-1]
        self.ps_selected_line_power = self.ps_ax_power.axvline(last_time, color='#98C379', linestyle='-', linewidth=2, zorder=25)
        self.ps_selected_line_freq = self.ps_ax_freq.axvline(last_time, color='#98C379', linestyle='-', linewidth=2, zorder=25)
        
        # (Hover and Pinned logic is identical to load_shedding_normal_case)
        self.pinned_annotation_ps = None
        self.ps_canvas.mpl_connect('button_press_event', self.on_click_percentage_shedding)
        
        self.update_percentage_shedding_table(last_time) # <--- เรียกฟังก์ชันใหม่
        self.ps_canvas.draw()

    def setup_summary_and_legend_item(self, name, color):
        hex_color = mcolors.to_hex(color); item_frame = ctk.CTkFrame(self.summary_panel, fg_color="transparent")
        item_frame.pack(fill="x", padx=10, pady=2); var = tk.BooleanVar(value=True)
        checkbox = ctk.CTkCheckBox(item_frame, text="", variable=var, width=20); checkbox.pack(side="left")
        ctk.CTkLabel(item_frame, text=f"■ {name}", text_color=hex_color, font=("Segoe UI", 12)).pack(side="left", padx=5)
        value_label = ctk.CTkLabel(item_frame, text="- MW", font=("Segoe UI", 12, "bold")); value_label.pack(side="right", padx=5)
        self.summary_labels[name] = {'checkbox': checkbox, 'label': value_label, 'color': hex_color, 'var': var}
    def toggle_line_visibility(self, name):
        line = self.plotted_lines[name]; var = self.summary_labels[name]['var']
        line.set_visible(var.get()); self.cont_canvas.draw_idle()
    def update_summary_panel(self, selected_time):
        if not self.interactive_plot_data: return
        summary_data = self.interactive_plot_data['summary_data']
        load_val = summary_data['total_load_mw'].get(selected_time, 0)
        self.summary_labels["Total Load"]['label'].configure(text=f"{load_val:,.2f} MW")
        pivoted_gens = summary_data['pivoted_gens_mw']
        for gen_id in pivoted_gens.columns:
            name = f'Gen {int(gen_id)}'
            if name in self.summary_labels:
                gen_val = pivoted_gens.loc[selected_time, gen_id]
                self.summary_labels[name]['label'].configure(text=f"{gen_val:,.2f} MW")
    
    def on_hover(self, event):
        is_visible = self.hover_line and self.hover_line.get_visible()
        if event.inaxes is self.cont_ax:
            nearest_time = self.find_nearest_time(event.xdata, "summary_data")
            if nearest_time is None: return
            self.hover_line.set_xdata([nearest_time])
            tooltip_text = f"Time: {nearest_time.strftime('%H:%M')}\n"; all_lines_with_names = list(self.plotted_lines.items())
            for i, (name, line) in enumerate(all_lines_with_names):
                point = self.hover_points[i]
                if line.get_visible():
                    y_val = np.interp(plt.matplotlib.dates.date2num(nearest_time), plt.matplotlib.dates.date2num(line.get_xdata()), line.get_ydata())
                    point.set_data([nearest_time], [y_val]); point.set_color(line.get_color()); point.set_visible(True)
                    if name == "Total Load": self.annotation.xy = (nearest_time, y_val)
                    tooltip_text += f"{name}: {y_val:.2f} MW\n"
                else: point.set_visible(False)
            self.annotation.set_text(tooltip_text.strip())
            if not is_visible:
                self.hover_line.set_visible(True); self.annotation.set_visible(True)
                self.cont_canvas.draw_idle()
        else:
            if is_visible:
                self.hover_line.set_visible(False); self.annotation.set_visible(False)
                for point in self.hover_points: point.set_visible(False)
                self.cont_canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes is self.cont_ax and event.button == 1:
            nearest_time = self.find_nearest_time(event.xdata, "summary_data")
            if nearest_time: self.selected_line.set_xdata([nearest_time]); self.update_summary_panel(nearest_time); self.update_interactive_table(nearest_time, self.cont_tree, self.cont_table_title_var); self.cont_canvas.draw_idle()
    
    def on_hover_iterative(self, event):
        visible = False
        if event.inaxes is self.iter_ax_power or event.inaxes is self.iter_ax_freq:
            nearest_time = self.find_nearest_time(event.xdata, "summary_data")
            if nearest_time:
                visible = True
                self.iter_hover_line_power.set_xdata([nearest_time])
                self.iter_hover_line_freq.set_xdata([nearest_time])
                load_val = self.interactive_plot_data['summary_data']['total_load_mw'].get(nearest_time, 0)
                self.iter_hover_point.set_data([nearest_time], [load_val])
                y_range = self.iter_ax_power.get_ylim()
                y_pos_norm = (load_val - y_range[0]) / (y_range[1] - y_range[0])
                offset = (20, 20) if y_pos_norm < 0.8 else (20, -60)
                self.iter_annotation.set_position(offset)
                self.iter_annotation.xy = (nearest_time, load_val)
                self.iter_annotation.set_text(f"Time: {nearest_time.strftime('%H:%M')}")
        
        if hasattr(self, 'iter_hover_line_power') and self.iter_hover_line_power and self.iter_hover_line_power.get_visible() != visible:
            self.iter_hover_line_power.set_visible(visible)
            self.iter_hover_line_freq.set_visible(visible)
            self.iter_hover_point.set_visible(visible)
            self.iter_annotation.set_visible(visible)
            self.iter_canvas.draw_idle()

    def on_click_iterative(self, event):
        ax = event.inaxes
        if (ax is self.iter_ax_power or ax is self.iter_ax_freq) and event.button == 1:
            nearest_time = self.find_nearest_time(event.xdata, "summary_data")
            if nearest_time: 
                self.selected_line_power.set_xdata([nearest_time]); self.selected_line_freq.set_xdata([nearest_time])
                self.update_explanation_panel(nearest_time)
                self.update_interactive_table(nearest_time, self.iter_tree, self.iter_table_title_var)
                
                if hasattr(self, 'pinned_annotation') and self.pinned_annotation: self.pinned_annotation.set_visible(False)
                
                summary = self.interactive_plot_data['summary_data']
                load_val = summary['total_load_mw'].get(nearest_time, 0); pg_val = summary['total_pg_mw'].get(nearest_time, 0)
                pmax_val = summary['microgrid_pmax']; freq_val = summary['frequency_series']['frequency'].get(nearest_time, 0)
                pin_text = (f"Time: {nearest_time.strftime('%H:%M')}\nFrequency: {freq_val:.3f} Hz\n------------------\n"
                            f"Total Load: {load_val:.2f} MW\nTotal Gen: {pg_val:.2f} MW\nMG Pmax: {pmax_val:.2f} MW")
                y_range = self.iter_ax_power.get_ylim(); y_pos_norm = (load_val - y_range[0]) / (y_range[1] - y_range[0])
                offset = (20, 20) if y_pos_norm < 0.6 else (20, -80)
                self.pinned_annotation = self.iter_ax_power.annotate(pin_text, xy=(nearest_time, load_val), xytext=offset, textcoords="offset points",
                                            bbox=dict(boxstyle="round,pad=0.5", fc="#282C34", ec="#98C379", lw=1, alpha=0.9),
                                            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color='white'),
                                            visible=True, zorder=40, fontname="Arial", fontsize=10)
                self.iter_canvas.draw_idle()
    
    def on_click_adaptive_shedding(self, event):
        ax = event.inaxes
        if (ax is self.as_ax_power or ax is self.as_ax_freq) and event.button == 1:
            nearest_time = self.find_nearest_time(event.xdata, "summary_data")
            if nearest_time: 
                self.as_selected_line_power.set_xdata([nearest_time])
                self.as_selected_line_freq.set_xdata([nearest_time])
                self.update_adaptive_shedding_table(nearest_time) # <--- เรียกฟังก์ชันใหม่
                
                # (Pinned Tooltip Logic - same as load_shedding_normal)
                if hasattr(self, 'pinned_annotation_as') and self.pinned_annotation_as: self.pinned_annotation_as.set_visible(False)
                summary = self.interactive_plot_data['summary_data']
                load_before = summary['total_load_mw_before'].get(nearest_time, 0); load_after = summary['total_load_mw_after'].get(nearest_time, 0)
                freq_before = summary['frequency_series_before']['freq_before'].get(nearest_time, 0); freq_after = summary['frequency_series_after']['freq_after'].get(nearest_time, 0)
                mw_shed = summary['mw_shed_series']['mw_shed'].get(nearest_time, 0)
                pin_text = (f"Time: {nearest_time.strftime('%H:%M')}\n"
                            f"Freq (Before): {freq_before:.3f} Hz\nFreq (After): {freq_after:.3f} Hz\n"
                            f"------------------\n"
                            f"Load (Before): {load_before:.2f} MW\nLoad (After): {load_after:.2f} MW\n"
                            f"MW Shed: {mw_shed:.2f} MW")
                y_range = self.as_ax_power.get_ylim(); y_pos_norm = (load_after - y_range[0]) / (y_range[1] - y_range[0])
                offset = (20, 20) if y_pos_norm < 0.6 else (20, -80)
                self.pinned_annotation_as = self.as_ax_power.annotate(pin_text, xy=(nearest_time, load_after), xytext=offset, textcoords="offset points",
                                            bbox=dict(boxstyle="round,pad=0.5", fc="#282C34", ec="#98C379", lw=1, alpha=0.9),
                                            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color='white'),
                                            visible=True, zorder=40, fontname="Arial", fontsize=10)
                self.as_canvas.draw_idle()

    def update_adaptive_shedding_table(self, selected_time):
        if not self.interactive_plot_data: return
        time_str = selected_time.strftime('%H:%M')
        self.as_table_title_var.set(f"ตารางบันทึกการตัดโหลด (Adaptive) ณ เวลา {time_str}")
        df = self.interactive_plot_data['shed_loads_df']
        
        self.clear_table(self.as_tree)
        if df.empty or 'datetime' not in df.columns:
            return
            
        df_at_time = df[df['datetime'] == selected_time]
        for i, (_, row) in enumerate(df_at_time.iterrows()):
            mva = np.sqrt(row['MW_Shed']**2 + row['MVAR_Shed']**2)
            # --- ส่วนที่แก้ไข: สร้าง String สำหรับ Priority ---
            priority_str = f"{row['Priority_Before']:.1f} → {row['Priority_After']:.1f}"
            values = [i+1, int(row['BusID']), priority_str, f"{row['Shed_Percent']:.1f}%", f"{row['MW_Shed']:.4f}", f"{mva:.4f}"]
            self.as_tree.insert("", "end", values=values)

    def on_hover_loadshedding(self, event):
        visible = False
        if event.inaxes is self.ls_ax_power or event.inaxes is self.ls_ax_freq:
            nearest_time = self.find_nearest_time(event.xdata, "summary_data")
            if nearest_time:
                visible = True
                self.ls_hover_line_power.set_xdata([nearest_time])
                self.ls_hover_line_freq.set_xdata([nearest_time])
                load_after = self.interactive_plot_data['summary_data']['total_load_mw_after'].get(nearest_time, 0)
                self.ls_hover_point.set_data([nearest_time], [load_after])
                y_range = self.ls_ax_power.get_ylim()
                y_pos_norm = (load_after - y_range[0]) / (y_range[1] - y_range[0])
                offset = (20, 20) if y_pos_norm < 0.8 else (20, -60)
                self.ls_annotation.set_position(offset)
                self.ls_annotation.xy = (nearest_time, load_after)
                self.ls_annotation.set_text(f"Time: {nearest_time.strftime('%H:%M')}")
        
        if hasattr(self, 'ls_hover_line_power') and self.ls_hover_line_power and self.ls_hover_line_power.get_visible() != visible:
            self.ls_hover_line_power.set_visible(visible)
            self.ls_hover_line_freq.set_visible(visible)
            self.ls_hover_point.set_visible(visible)
            self.ls_annotation.set_visible(visible)
            self.ls_canvas.draw_idle()

    def on_click_loadshedding(self, event):
        ax = event.inaxes
        if (ax is self.ls_ax_power or ax is self.ls_ax_freq) and event.button == 1:
            nearest_time = self.find_nearest_time(event.xdata, "summary_data")
            if nearest_time: 
                self.ls_selected_line_power.set_xdata([nearest_time])
                self.ls_selected_line_freq.set_xdata([nearest_time])
                self.update_loadshedding_table(nearest_time)
                if hasattr(self, 'pinned_annotation_ls') and self.pinned_annotation_ls: self.pinned_annotation_ls.set_visible(False)
                summary = self.interactive_plot_data['summary_data']
                load_before = summary['total_load_mw_before'].get(nearest_time, 0); load_after = summary['total_load_mw_after'].get(nearest_time, 0)
                freq_before = summary['frequency_series_before']['freq_before'].get(nearest_time, 0); freq_after = summary['frequency_series_after']['freq_after'].get(nearest_time, 0)
                mw_shed = summary['mw_shed_series']['mw_shed'].get(nearest_time, 0)
                pin_text = (f"Time: {nearest_time.strftime('%H:%M')}\n"
                            f"Freq (Before): {freq_before:.3f} Hz\nFreq (After): {freq_after:.3f} Hz\n"
                            f"------------------\n"
                            f"Load (Before): {load_before:.2f} MW\nLoad (After): {load_after:.2f} MW\n"
                            f"MW Shed: {mw_shed:.2f} MW")
                y_range = self.ls_ax_power.get_ylim(); y_pos_norm = (load_after - y_range[0]) / (y_range[1] - y_range[0])
                offset = (20, 20) if y_pos_norm < 0.6 else (20, -80)
                self.pinned_annotation_ls = self.ls_ax_power.annotate(pin_text, xy=(nearest_time, load_after), xytext=offset, textcoords="offset points",
                                            bbox=dict(boxstyle="round,pad=0.5", fc="#282C34", ec="#98C379", lw=1, alpha=0.9),
                                            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color='white'),
                                            visible=True, zorder=40, fontname="Arial", fontsize=10)
                self.ls_canvas.draw_idle()

    def on_click_percentage_shedding(self, event):
        ax = event.inaxes
        if (ax is self.ps_ax_power or ax is self.ps_ax_freq) and event.button == 1:
            nearest_time = self.find_nearest_time(event.xdata, "summary_data")
            if nearest_time: 
                self.ps_selected_line_power.set_xdata([nearest_time])
                self.ps_selected_line_freq.set_xdata([nearest_time])
                self.update_percentage_shedding_table(nearest_time) # <--- เรียกฟังก์ชันใหม่
                
                # (Pinned Tooltip Logic - same as load_shedding_normal)
                if hasattr(self, 'pinned_annotation_ps') and self.pinned_annotation_ps: self.pinned_annotation_ps.set_visible(False)
                summary = self.interactive_plot_data['summary_data']
                load_before = summary['total_load_mw_before'].get(nearest_time, 0); load_after = summary['total_load_mw_after'].get(nearest_time, 0)
                freq_before = summary['frequency_series_before']['freq_before'].get(nearest_time, 0); freq_after = summary['frequency_series_after']['freq_after'].get(nearest_time, 0)
                mw_shed = summary['mw_shed_series']['mw_shed'].get(nearest_time, 0)
                pin_text = (f"Time: {nearest_time.strftime('%H:%M')}\n"
                            f"Freq (Before): {freq_before:.3f} Hz\nFreq (After): {freq_after:.3f} Hz\n"
                            f"------------------\n"
                            f"Load (Before): {load_before:.2f} MW\nLoad (After): {load_after:.2f} MW\n"
                            f"MW Shed: {mw_shed:.2f} MW")
                y_range = self.ps_ax_power.get_ylim(); y_pos_norm = (load_after - y_range[0]) / (y_range[1] - y_range[0])
                offset = (20, 20) if y_pos_norm < 0.6 else (20, -80)
                self.pinned_annotation_ps = self.ps_ax_power.annotate(pin_text, xy=(nearest_time, load_after), xytext=offset, textcoords="offset points",
                                            bbox=dict(boxstyle="round,pad=0.5", fc="#282C34", ec="#98C379", lw=1, alpha=0.9),
                                            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color='white'),
                                            visible=True, zorder=40, fontname="Arial", fontsize=10)
                self.ps_canvas.draw_idle()

    def find_nearest_time(self, event_x, data_key):
        if not self.interactive_plot_data or event_x is None: return None
        summary_dict = self.interactive_plot_data.get(data_key, {})
        time_series = None
        if 'total_load_mw' in summary_dict:
            time_series = summary_dict['total_load_mw'].index
        elif 'total_load_mw_before' in summary_dict:
            time_series = summary_dict['total_load_mw_before'].index
        else:
            return None
        time_as_float = plt.matplotlib.dates.date2num(time_series); idx = np.abs(time_as_float - event_x).argmin()
        return time_series[idx]

    def update_interactive_table(self, selected_time, tree_widget, title_var):
        if not self.interactive_plot_data: return
        time_str = selected_time.strftime('%H:%M')
        base_title = f"ตารางแสดงผลการรัน loadflow ณ เวลา {time_str}"; df = self.interactive_plot_data['full_df']; df_at_time = df[df['datetime'] == selected_time]
        warning_msg = ""
        if 'Warning' in df_at_time.columns and not df_at_time['Warning'].empty:
            first_warning = df_at_time['Warning'].iloc[0]
            if first_warning: warning_msg = f" ({first_warning})"
        title_var.set(base_title + warning_msg)
        self.clear_table(tree_widget)
        for _, row in df_at_time.iterrows():
            cols = list(tree_widget["columns"])
            values = []
            for col in cols:
                if col in row:
                    if col in ['BusID', 'Type']: values.append(f"{int(row[col])}")
                    else: values.append(f"{row[col]:.4f}")
                else: values.append("")
            tree_widget.insert("", "end", values=values)

    def update_loadshedding_table(self, selected_time):
        if not self.interactive_plot_data: return
        time_str = selected_time.strftime('%H:%M')
        self.ls_table_title_var.set(f"ตารางบันทึกการตัดโหลด (Load Shedding Log) ณ เวลา {time_str}")
        df = self.interactive_plot_data['shed_loads_df']
        
        self.clear_table(self.ls_tree)
        if df.empty or 'datetime' not in df.columns:
            return
            
        df_at_time = df[df['datetime'] == selected_time]
        for i, (_, row) in enumerate(df_at_time.iterrows()):
            mva = np.sqrt(row['MW_Shed']**2 + row['MVAR_Shed']**2)
            values = [i+1, int(row['BusID']), int(row['Priority']), f"{row['MW_Shed']:.4f}", f"{mva:.4f}"]
            self.ls_tree.insert("", "end", values=values)

    def update_percentage_shedding_table(self, selected_time):
        if not self.interactive_plot_data: return
        time_str = selected_time.strftime('%H:%M')
        self.ps_table_title_var.set(f"ตารางบันทึกการตัดโหลด (Percentage) ณ เวลา {time_str}")
        df = self.interactive_plot_data['shed_loads_df']
        
        self.clear_table(self.ps_tree)
        if df.empty or 'datetime' not in df.columns:
            return
            
        df_at_time = df[df['datetime'] == selected_time]
        for i, (_, row) in enumerate(df_at_time.iterrows()):
            mva = np.sqrt(row['MW_Shed']**2 + row['MVAR_Shed']**2)
            values = [i+1, int(row['BusID']), int(row['Priority']), f"{row['Shed_Percent']:.1f}%", f"{row['MW_Shed']:.4f}", f"{mva:.4f}"]
            self.ps_tree.insert("", "end", values=values)

    def update_explanation_panel(self, selected_time):
        explanation_text = self._generate_frequency_explanation(selected_time)
        self.explanation_textbox.configure(state="normal"); self.explanation_textbox.delete("1.0", "end")
        self.explanation_textbox.insert("1.0", explanation_text); self.explanation_textbox.configure(state="disabled")

    def _generate_frequency_explanation(self, selected_time):
        if not self.interactive_plot_data or not self.current_system_data: return "No data to generate explanation."
        summary = self.interactive_plot_data['summary_data']
        params = self.interactive_plot_data['calculation_params']
        f_nominal = params['base_freq']
        total_load = summary['total_load_mw'].get(selected_time, 0)
        total_gen = summary['total_pg_mw'].get(selected_time, 0)
        delta_p_calc = total_load - total_gen
        final_freq = summary['frequency_series']['frequency'].get(selected_time, f_nominal)
        power_imbalance_from_summary = summary['power_imbalance_series']['power_imbalance'].get(selected_time, 0)
        online_dgs = params['online_dgs']
        text = f"=============== INPUT PARAMETERS @ {selected_time.strftime('%H:%M')} ===============\n\n"
        text += f"  System Base MVA       : {params['base_mva']:.2f}\n"
        text += f"  System Base Frequency : {f_nominal:.2f} Hz\n"
        text += f"  Power Imbalance (ΔP)  : {power_imbalance_from_summary:.4f} MW\n\n"
        text += "--- Active Microgrid Generator Parameters ---\n"
        gen_table_data = [[int(gen['GenID']), gen['Pmax_MW'], gen['Droop_R']] for _, gen in online_dgs.iterrows()]
        text += tabulate(gen_table_data, headers=["GenID", "Pmax (MW)", "Droop (p.u.)"], tablefmt="grid") + "\n\n"
        text += f"=============== CALCULATION STEPS @ {selected_time.strftime('%H:%M')} ===============\n\n"
        text += "1. การคำนวณกำลังไฟฟ้าไม่สมดุล (Power Imbalance)\n"
        text += f"   - Total Load (P_load) = {total_load:.4f} MW\n"
        text += f"   - Total Generation (P_gen) = {total_gen:.4f} MW\n"
        text += f"   - สมการ: ΔP = ΣP_load - ΣP_gen = {delta_p_calc:.4f} MW\n"
        text += f"   *Note: Power Imbalance ที่ใช้คำนวณความถี่ (ΔP) คือ {power_imbalance_from_summary:.4f} MW\n\n"
        
        inv_r_sum = 0
        for _, gen in online_dgs.iterrows():
            r_pu = gen['Droop_R']; p_rated = gen['Pmax_MW']
            if p_rated > 0: inv_r_sum += (1 / ((r_pu * f_nominal) / p_rated))
        r_sys = 1 / inv_r_sum if inv_r_sum > 0 else float('inf')
        text += "2. การคำนวณ Droop สมมูลของระบบ (System Equivalent Droop)\n"
        text += f"   - R_sys = ( Σ(1 / R_i) )^-1 = {r_sys:.6f} Hz/MW\n\n"
        delta_f = -r_sys * power_imbalance_from_summary
        text += "3. การคำนวณความเบี่ยงเบนของความถี่ (Frequency Deviation)\n"
        text += f"   - สมการ: Δf = -R_sys * ΔP = -{r_sys:.6f} * {power_imbalance_from_summary:.4f} = {delta_f:.4f} Hz\n\n"
        f_final_calc = f_nominal + delta_f
        text += "4. การคำนวณหาความถี่สุดท้าย (Final Settled Frequency)\n"
        text += f"   - สมการ: f_final = f_nominal + Δf = {f_nominal:.2f} + ({delta_f:.4f}) = {f_final_calc:.4f} Hz\n"
        text += f"   - (ค่าที่แสดงบนกราฟ: {final_freq:.4f} Hz)\n"
        return text

    def clear_table(self, tree_widget):
        for i in tree_widget.get_children(): tree_widget.delete(i)
    def save_results_to_file(self):
        if self.last_results_data is None: return
        filename = self.filename_entry.get()
        if not filename: return
        if not filename.lower().endswith('.csv'): filename += '.csv'
        filepath = os.path.join(self.RESULTS_PATH, filename)
        try:
            if isinstance(self.last_results_data, dict) and 'full_df' in self.last_results_data:
                self.last_results_data['full_df'].to_csv(filepath, index=False)
            elif 'shed_loads_df' in self.last_results_data:
                # Save multiple dataframes to different sheets in an Excel file
                if not filename.lower().endswith('.xlsx'):
                    filename = os.path.splitext(filename)[0] + '.xlsx'
                filepath = os.path.join(self.RESULTS_PATH, filename)
                with pd.ExcelWriter(filepath) as writer:
                    self.last_results_data['full_df'].to_csv(writer, sheet_name='Loadflow_Results', index=False)
                    self.last_results_data['shed_loads_df'].to_csv(writer, sheet_name='Loadshed_Log', index=False)
            else: raise TypeError("Result data is not in a saveable format.")
            
            self.save_button.configure(text="Saved!", fg_color="green")
            self.after(2000, lambda: self.save_button.configure(text="Save Results", fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"]))
        except Exception as e:
            messagebox.showerror("Save Error", f"An error occurred while saving:\n{e}")
            self.save_button.configure(text="Error!", fg_color="red")
            self.after(2000, lambda: self.save_button.configure(text="Save Results", fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"]))

if __name__ == '__main__':
    app = App()
    app.mainloop()