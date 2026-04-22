# Burgers PINN Project

A local scientific machine learning project for learning a physics-informed neural network surrogate on the 1D Burgers equation.

## Goals

- Build a baseline PINN model
- Train locally
- Visualize training and prediction results
- Evaluate the model with a simplified reference solution
- Test simple sampling and active-learning-inspired enrichment strategies

## What This Project Is For

This project is a small scientific machine learning prototype built around the 1D Burgers equation.

Its purpose is not simply to obtain the best possible numerical solution, but rather to build and test a full workflow for PINN-based surrogate modeling.

More specifically, the project is used to:

- learn how to build a physics-informed neural network (PINN),
- approximate the solution of a PDE with a neural network surrogate,
- evaluate the model with visualizations and error metrics,
- explore how different sampling strategies affect PINN performance.

In this sense, the project is a controlled and simplified environment for testing ideas that are relevant to scientific computing and AI-based acceleration.

## What the Model Is Actually Learning

The model is learning a function that maps space-time coordinates to the PDE solution value.

More precisely, the neural network is trained to approximate:

u(x, t)

where:
- `(x, t)` is the input,
- `u(x, t)` is the solution of the 1D Burgers equation.

This means that the model is not doing classification or standard tabular regression.
Instead, it is learning a continuous function over the space-time domain.

The training objective combines three types of constraints:

- the PDE residual,
- the boundary conditions,
- the initial condition.

So the model is trained not only to fit values, but also to remain consistent with the underlying physical equation.

## Knowledge Involved

This project combines several areas of knowledge:

### Mathematics and PDEs
- Burgers equation
- initial condition
- boundary condition
- PDE residual
- space-time solution representation

### Scientific Computing
- approximation of PDE solutions
- reference solution
- quantitative error evaluation
- MSE and Relative L2 Error
- sampling strategy design

### Machine Learning and Deep Learning
- neural networks
- loss functions
- optimization with Adam
- model training and inference
- function approximation

### Scientific Machine Learning
This is the main perspective of the project:
the project combines physical constraints and neural networks in order to build a surrogate model for a PDE solution.

## Why This Project Matters

The main value of this project is methodological.

It provides:

- a baseline PINN implementation,
- an evaluation workflow,
- visualization tools,
- experiments on different sampling strategies,
- a small but meaningful scientific ML testbed.

Even though the current setup is simplified, the project already shows that sampling strategy can strongly affect PINN performance.

## Connection with the ANEO Internship Direction

This project is not the ANEO internship topic itself, but it is strongly related to the same technical direction.

The connection is the following:

- in both cases, the idea is to use AI models as surrogate approximations for expensive scientific computations,
- in both cases, error and stability matter, not only speed,
- in both cases, the goal is to better understand when learned approximations can help scientific workflows.

This project can therefore be seen as a small prototype aligned with the broader ANEO direction:
combining scientific computing, AI-based approximation, and evaluation of accuracy/stability trade-offs.


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

## Why This Method Was Chosen

PINNs are attractive because they allow the integration of physical knowledge directly into the training process.

Instead of relying only on labeled data, the model can be trained using:

- the governing PDE,
- boundary conditions,
- initial conditions.

This makes PINNs a natural starting point for learning PDE surrogates in a scientific machine learning setting.

## Why the Current Active Enrichment Did Not Work Well

In the current simplified setup, the active enrichment strategies did not improve the final error.

Possible reasons include:

- the simplified reference solution is not a full high-fidelity Burgers solver,
- PINN training is sensitive to the sampling distribution,
- adding more collocation points does not provide direct supervised labels,
- local enrichment may over-focus on specific regions and reduce global stability.

This negative result is still meaningful:
it shows that active sampling is not automatically beneficial for PINNs, and that stable uniform sampling can remain a strong baseline.

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

