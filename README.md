# MarineVesselModels

using:

```
https://github.com/VectorWang2015/USVWidgets
```

## Records

> Fossen_zigzag_xx_yy_zz_tt.npy

Fossen model, zigzag case with target psi xx degrees; base thrust xx N for each thrust; delta thrust for turing zz N; time step tt s.  

data: 9xn; [x, y, psi, u, v, r, tau1, tau2, tau3]  

> Fossen_nps_zigzag_xx_yy_zz_tt.npy

Fossen model with thrust model using nps as each thruts input, zigzag case with target psi xx degrees; base nps xx for each thrust; delta nps for turing zz N; time step tt s.  

data: 8xn; [x, y, psi, u, v, r, left_nps, right_nps]  

> KT_zigzag_xx_yy_tt.npy

First order non-linear response model, zigzag case with target psi xx degrees; turning rudder angle delta yy degrees; time step tt s;  

data: 7xn; [x, y, psi, u, v, r, delta]  