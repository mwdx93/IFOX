# IFOX
Improved FOX optimization algorithm (IFOX)

This repository contains a Mealpy‑compatible implementation of the IFOX algorithm,  
which mimics red‑fox hunting behavior of the oringial FOX algorithm with fitness‑based adaptive exploration & exploitation.

## Installation
```bash
git clone https://github.com/mwdx93/IFOX.git
cd IFOX
pip install mealpy
```

## Usage
```python
import IFOX
from mealpy.problems import Sphere
algo = IFOX(epoch=1000, pop_size=50)
best = algo.solve(Sphere())
print(best)
```

## How to cite
If you use IFOX please cite:

- Jumaah, M. A., Ali, Y. H., Rashid, T. A., & Vimal, S. (2024). FOXANN: A Method for Boosting Neural Network Performance. Journal of Soft Computing and Computer Applications, 1(1), Article 1001. https://doi.org/10.70403/3008-1084.1001 
- Jumaah, M. A., Ali, Y. H., & Rashid, T. A. (2025). Efficient Q‑learning Hyperparameter Tuning Using FOX Optimization Algorithm. Results in Engineering, 25, 104341. https://doi.org/10.1016/j.rineng.2025.104341 

```bibtex
@article{jumaah2024FOXANN,
  title   = {FOXANN: A Method for Boosting Neural Network Performance},
  author  = {Jumaah, Mahmood A. and Ali, Yossra H. and Rashid, Tarik A. and Vimal, S.},
  journal = {Journal of Soft Computing and Computer Applications},
  volume  = {1},
  number  = {1},
  pages   = {Article\,1001},
  year    = {2024},
  doi     = {10.70403/3008-1084.1001}
}

@article{jumaah2025EfficientQFOX,
  title   = {Efficient Q‑learning Hyperparameter Tuning Using FOX Optimization Algorithm},
  author  = {Jumaah, Mahmood A. and Ali, Yossra H. and Rashid, Tarik A.},
  journal = {Results in Engineering},
  volume  = {25},
  pages   = {104341},
  year    = {2025},
  doi     = {10.1016/j.rineng.2025.104341}
}
