# NucleonNucleon

This repository contains the scripts used to perform the numerical calculations of the master's thesis ''Nucleon-nucleon potentials from generalized Skyrme models''.

Below is a brief description of the procedure:

## Step 1: generate (r,Q)
- Use _sample.py_. Generates _Q.npy_,_r.npy_ and _uvw.npy_.

## Step 2: compute volume forms
- Use _volume_form.py_. Generates _dQ.npy_. Imports _uvw.npy_

## Step 3: obtain the radial function
- Use _hedgehog_1S.nb_ to solve the differential equation for each model. Generates _data_sfX.txt_.
- Use _interp.py_ to interpolate _f(r)_ in the values of _r_ that will be needed later (missing here).

## Step 4: compute the derivatives
- Use _deriv_X.py_ to compute the derivatives (missing here).

## Step 5: compute the metric
- Use _metric.nb_ to expand the terms of the metric. Copy each term into a separate Python function: _gX.py_.
- Use _metric.py_ to compute the metric (missing here). Imports _gX.py_.

## Step 6: compute the potential
- Not chekced yet, but workflow similar to Step 4. 

## Step 7: compute the Fourier coefficients
- Use _fourier.py_ to compute the Fourier coefficients. 
