# NucleonNucleon

This repository contains the scripts used to perform the numerical calculations of the master thesis ''Nucleon-nucleon potentials from generalized Skyrme models''.

Below is a brief description of the purpose of each script:

- _hedgehog_1S.nb_ - Solve the field equations for different Skyrme models to obtain the profile functions.
- _sample.py_ - Generate the (r,Q) values.
- _interp.py_ - Interpolate the profile function on the points needed subsequently.
- _prod_read.py_ - Generate a field given by the product approximation.
- _deriv_y.py_ - Generate the derivatives of the field w.r.t (y1,y2,y3).
- _deriv_x.py_ - Generate the derivatives of the field w.r.t (x1,x2,x3).
- _deriv_Q.py_ - Generate the derivatives of the field w.r.t (Q1,Q2).
- _metric.py_ - Generate the components of the metric.
- _fourier.py_ - Generate the Fourier coefficients.

Note that some scripts use data files produced by other scripts. The dependencies are explained in _dep.png_


