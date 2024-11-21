# petlab

## An open-source framework for Closed-Loop Reservoir Management (CLRM). 

As in CLRM, the two main frameworks featured in this tool are the History matching framework and the Reservoir optimization framework. However, this tool also has support features such as generation of geological realizations and extraction of simulation model results. There are 5 main modules: 

1. Create ensemble : creating an ensemble by populating the values based on some distributions.  
2. Run ensemble : running the ensemble 
3. Extract ensemble : extracting the result from the ensemble that has been run
4. Optimize ensemble : finding the most optimal control based on some objective function
5. History match ensemble : updating the ensemble to match the historical data 



## History matching framework

History match methods:
1. Ensemble Smoother with Multiple Data Assimilation (ESMDA), 
2. (New method) Partial Least Square Regression with Spectral Decomposition. 

## Optimization framework

Optimization methods:
1. Trust-region method with SQP-Filter
2. COBYLA
3. NOMAD

## Configuring a study



<details><summary> <b> flash_calculation.py </b> </summary>

Calculate liquid and gas composition (flash calculation) given fluid composition, pressure and temperature condition. (Now it works only on field unit)

Run flash_calculation.py with the following command:

```python flash_calculation.py /path/to/fluid_dataset.json/ /path/to/result.csv/```

Sample data set is given under the json_files folder. Run the sample data as follows:

``` python flash_calculation.py ./json_files/flash_calculation/pvt_dataset_2.json ./results/pvt_dataset_2_result.csv ```

The .JSON file must include the following:
``` 
  "PressureK" : Convergence pressure (psi) 
  "Pressure" : Pressure condition for flash calculation (psi)
  "Temperature" : Temperature condition for flash calculation (Rankine)
  "A0" : A0 variable
  "fv" : Initial condition for fv
  "max_iter" : Maximum number of iteration to get fv
  "Component" : {
    "<Component Name 1>" : {
      "Mole_Fraction" : total component fraction
      "Critical_Pressure" : critical pressure of the component (psia)
      "Critical_Temperature" : critical temperature of the component (psia) 
      "Accentric_Factor" : accentric factor of the component
      }
    }
```

</details>

<details><summary> <b> drawdown_test.py </b> </summary>

Calculate the permeability of a reservoir given a data set of drawdown test (pressure and time). (Now it works only on SI unit)

Run drawdown_test.py with the following command:

```python flash_calculation.py /path/to/drawdown_dataset.json/ /path/to/result.csv/```

Sample data set is given under the json_files folder. Run the sample data as follows:

``` python flash_calculation.py ./json_files/dradown_test/dd_dataset_1.json ./results/dd_dataset_1_result.csv ```

The .JSON file must include the following:
``` 
  "Rate" : constant fluid production rate (m3/s)
  "Viscosity" : fluid viscosity (Pa s)
  "Porosity" : Average reservoir porosity (fraction)
  "Initial_Pressure" : Initial pressure condition (Pa)
  "Well_Radius" : Radius of the production well (m)
  "Reservoir_Height" : Average height radius of the reservoir (h)
  "Total_Compressibility" : Average total compressibility (1/Pa) 
  "Pressure_Data" : List of well pressure data (Pa)
  "Time_Data" : List of the corresponding time data (s)
```
</details>


<details><summary> <b> npv_calculation.py </b> </summary>

NPV Calculation given production data, field development plan, and business expenditure parameters.  

Run npv_calculation.py with the following command:

```python npv_calculation.py /path/to/npv_dataset.json/ /path/to/result.csv/```

Sample data set is given under the json_files folder. Run the sample data as follows:

``` python npv_calculation.py ./json_files/npv_calculation/npv_calculation_dataset_1.json ./results/npv.csv ```

The .JSON file must include the following:
``` 
  "Years" : Number of years from where the first expenditure is made until the end of the analysis (Years)
  "Uptime" : The fraction of productive days in a year (fraction)
  "CAPEX" : {
    "Start_year" : The first year CAPEX is spent. Must be between 1 - Years
    "Fraction" : The list of how much fraction of CAPEX is being paid per year. e.g. [0.4, 0.6] represents spending 0.4 fraction of CAPEX in the first year and 0.6 fraction of CAPEX in the second year.
    "Amount" : The total amount of CAPEX (USD)
  }
 "OPEX" : {
  "Start_year : The first year OPEX is spent. Must be between 1 - Years
  "Amount" : Amount of OPEX spent per year (USD)
  }
 "DRILLEX" : {
   "Start_year" : The first year DRILLEX is spent (First well is drilled). Must be between 1 - Years
   "Excalation" : Exponential rate for the DRILLEX per year (fraction/Year)
   "Amount" : Base price for a well (USD/well)
   "Wells" : A list of numbers of well drilled per year e.g. [3, 5, 6, 6] represents drilling 3 wells in the first year, followed by 5 wells in the second year, followed by 6 wells in the third and fourth year.
 "Discount_rate" Discount rate (fraction) 
 "Oil_price" : Average oil price throughout the field development (USD/stb)
 "FOPT" : {
  "Start_year" : The first year oil is produced,
  "Data" : Numpy file of 1D production data, with each element represents total production per year
```
</details>
