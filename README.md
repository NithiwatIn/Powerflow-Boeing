# Powerflow-Boeing
A custom implementation of power flow analysis using the Newton-Raphson method, developed from scratch without external libraries. This project serves as a foundation for exploring and validating theoretical concepts in my doctoral research.

Directory Architecture

    Project Structure
        powerflow/
            │
            ├── data/                        # Data input files (bus, line, generator)
            │   ├── ieee14/
            │   │   ├── bus_data.csv
            │   │   ├── line_data.csv
            │   │   ├── generator_data.csv
            │   │   ├── load_data.csv
            │   │   └── system_config.csv
            │   ├── ieee30/
            │   │   └── ...
            │   └── ...
            │
            ├── models/                      # Structure data (Entities)
            │   ├── bus.py
            │   ├── line.py
            │   ├── generator.py
            │   └── system.py
            │
            ├── algorithms/                  # Power Flow Solver
            │   ├── newton_raphson.py
            │   ├── fast_decoupled.py
            │   └── gauss_seidel.py
            │
            ├── core/                        # Core Logic 
            │   ├── admittance.py            # Create Y-bus
            │   ├── mismatch.py              # Calculate P/Q mismatch
            │   ├── jacobian.py              # Create Jacobian
            │   ├── update_state.py          # Update V, θ
            │   └── frequency.py             # Calculate frequency in the system
            │
            ├── utils/                       # Utilities
            │   ├── file_io.py               # Load/Save CSV, JSON
            │   ├── plot.py                  # Plot output graph
            │   └── formatter.py             # Output formatting
            │
            ├── results/                     # Output data
            │   ├── ieee14_run1.csv
            │   └── plots/
            │
            ├── tests/                       # Unit tests
            │   ├── test_newton.py
            │   ├── test_ybus.py
            │   └── ...
            │
            ├── main.py                      # Software entry point
            └── README.md

folder data
    folder ieee-30
        - bus_data.csv
        - line_data.csv
        - generator_data.csv
        - load_data.csv
        - system_config.csv
folder simulation
    - newtonrapson_loadflow.py
folder utils
    - ybus_builder.py
    - data_manager.py