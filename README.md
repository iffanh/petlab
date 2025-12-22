# petlab

## An open-source framework for Closed-Loop Reservoir Management (CLRM). 

As in CLRM, the two main frameworks featured in this tool are the History matching framework and the Reservoir optimization framework. However, this tool also has support features such as generation of geological realizations and extraction of simulation model results. There are 5 main modules: 

1. Create ensemble : creating an ensemble by populating the values based on some distributions.  
2. Run ensemble : running the ensemble 
3. Extract ensemble : extracting the result from the ensemble that has been run
4. Optimize ensemble : finding the most optimal control based on some objective function
5. History match ensemble : updating the ensemble to match the historical data 
6. Evaluate ensemble : running an ensemble with a different set of controls


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

## Installation requirement

```shell
pip install gstools 
pip install tqdm 
pip install ecl
pip install wrapt_timeout_decorator
```