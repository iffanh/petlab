# petlab


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
