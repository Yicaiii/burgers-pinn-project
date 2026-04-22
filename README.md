# Burgers PINN Project

A local scientific machine learning project for learning a physics-informed neural network surrogate on the 1D Burgers equation.

## Goals

- Build a baseline PINN model
- Train locally
- Visualize training and prediction results
- Evaluate the model with a simplified reference solution
- Test simple sampling and active-learning-inspired enrichment strategies

## Project Structure

- `src/`: source code
- `data/`: datasets
- `outputs/`: figures and checkpoints
- `notebooks/`: experiments

## Current Progress

- Trained a baseline PINN on the 1D Burgers equation
- Saved the trained model checkpoint
- Generated a training loss curve
- Generated a prediction heatmap
- Generated prediction curves for several time slices
- Added evaluation with a simplified reference solution
- Computed quantitative metrics such as MSE and Relative L2 Error
- Tested two simple active-learning-inspired sampling strategies
- Compared several sampling strategies with summary plots

## Main Scripts

- `src/train.py`: train the baseline PINN model
- `src/evaluate.py`: evaluate the trained model and compare it with a simplified reference solution
- `src/train_active.py`: first active-learning-style enrichment experiment
- `src/train_active_v2.py`: mixed enrichment experiment with error-guided and random sampling
- `src/compare_results.py`: compare the results of the tested strategies

## How to Run

### 1. Activate the virtual environment

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
````

### 2. Train the baseline model

```powershell
python src/train.py
```

### 3. Evaluate the baseline model

```powershell
python src/evaluate.py
```

### 4. Run active-learning-style experiments

```powershell
python src/train_active.py
python src/train_active_v2.py
```

### 5. Compare experimental results

```powershell
python src/compare_results.py
```

## Results

The following sampling strategies were tested for the 1D Burgers PINN project:

1. Uniform baseline evaluation
2. Active Learning V1: pure error-guided enrichment
3. Active Learning V2: mixed enrichment (error-guided + random)

### Quantitative Results

| Method           |          MSE | Relative L2 Error | Observation                                  |
| ---------------- | -----------: | ----------------: | -------------------------------------------- |
| Baseline Eval    | 1.317788e-01 |      5.224842e-01 | Baseline PINN evaluation                     |
| Active V1 Before | 1.065557e-01 |      4.708752e-01 | Initial training before enrichment           |
| Active V1 After  | 1.213141e-01 |      5.024271e-01 | Slight degradation after enrichment          |
| Active V2 Before | 1.009018e-01 |      4.582125e-01 | Best pre-enrichment result among active runs |
| Active V2 After  | 1.699038e-01 |      5.945918e-01 | Clear degradation after mixed enrichment     |

## Interpretation

In the current simplified setup, active enrichment of collocation points did not improve the final error.

Both the pure error-guided strategy and the mixed strategy led to worse final performance after enrichment.

This suggests that, for this simplified Burgers PINN setting:

* uniform sampling remains the most stable strategy,
* local enrichment may over-focus on specific regions,
* changing the collocation distribution can destabilize PINN training.

## Current Conclusion

At this stage, the project shows that sampling strategy has a strong impact on PINN performance.

A simple active-learning approach is not automatically beneficial, and stable uniform sampling remains a strong baseline.

## Next Steps

Possible next improvements:

* use a more realistic numerical reference solver,
* improve path handling in all scripts,
* test different PINN hyperparameters,
* compare more sampling strategies,
* extend the project to a more robust scientific ML benchmark.

````

