
Python scripts accompanying the paper: <span style="color:lightblue">"Insights on the kinetic mechanism of manganese oxide precipitation from Mn2+ solutions using NaClO under highly acidic conditions, via experimental validation and numerical ODE fitting".</span>


## Requirements:

- [Python](https://www.python.org/) (version 3.6 or higher)
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [pandas](https://pandas.pydata.org/)

## Installing dependencies

It is recommended to use a package manager for Python. For example, using 
[conda](https://docs.conda.io/en/latest/#). Once installed, it is possible to
create a new environment to install the dependencies. For example:

```bash
   conda create -n <name> python=3.6 numpy scipy pandas
   conda activate <name>
```
Where you can replace `<name>` with the desired name for your environment.

## Running the script

After the dependencies are installed and the Python environment is activated,
navigate to the directory where the files `Mn_oxidation_slow_HOCl.py` and `Mn_oxidation_fast_HOCl.py`
are located. Once there, it is possible to run the script file by using:

```bash
    python Mn_oxidation_slow_HOCl.py
```
or
```bash
    python Mn_oxidation_fast_HOCl.py
```
depending on which HOCl addition speed is being modeled.

Once started, the script will look for the comma-separated values (.csv) files containing the experimental pH data (i.e. `pH_experimental_data_slow.csv` or `pH_experimental_data_fast.csv`), and then print the parameter results for the [H+] function fitting.
These parameters are only used as values for the integration of H+ concentration change over time
within the scope of the ODE system, and are only shown for information.

## Output

The script returns a comma-separated values (.csv) file, which then can be opened
using traditional spreadsheet software (such as Microsoft Excel, Google Sheets, 
Libreoffice Calc, etc.). Also, depending on the HOCl addition speed, the name of the file is
either `Mn_oxidation_slow_HOCl_concentrations.csv`, or `Mn_oxidation_fast_HOCl_concentrations.csv`.
The output file contains the concentrations (in mol/L) for each chemical component of the system over time,
namely:
- Mn2+ 
- HOCl
- H+