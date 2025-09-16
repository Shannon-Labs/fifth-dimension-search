# 🌌 The Fifth Dimension Hunt: Testing Kaluza-Klein Gravity with LIGO/Virgo

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CUDA/MPS Support](https://img.shields.io/badge/accelerated-CUDA%2FMPS-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **What if gravity can escape into a hidden dimension that we can't see?**

We're using LIGO/Virgo gravitational wave detectors to search for evidence of a 5th spatial dimension predicted by Kaluza-Klein theory. When black holes or neutron stars collide, they create ripples in spacetime. If there's a hidden dimension, these ripples would "leak" slightly into the extra dimension, changing the signal in a way we can measure.

## 🔬 The Core Discovery

Our simulations reveal that black hole mergers and neutron star mergers respond differently to a potential 5th dimension:

- **Black Holes** (pure spacetime): Show ~3 milliradian phase shifts
- **Neutron Stars** (matter): Show different phase evolution patterns  
- **The Smoking Gun**: This differential response could reveal extra dimensions

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/shannon-labs/dimension-search.git
cd dimension-search

# Install dependencies
pip install -r requirements.txt

# Run a simulation
python run/bssn_kk_evolver.py --test-mode

# Analyze results
python tools/analyze_waveforms.py
```

## 📊 The Physics: Kaluza-Klein Theory

In 1921, Theodor Kaluza showed that Einstein's General Relativity in 5 dimensions naturally unifies gravity and electromagnetism. The 5th dimension is "compactified" - curled up so small we can't see it directly.

### Key Parameters:
- `q`: KK charge - coupling to extra dimension (10⁻⁴ to 10⁻³)
- `L₅`: Compactification radius - size of hidden dimension
- `m₅`: 5D mass parameter - energy scale of effects

## 🎯 Current Status

⚠️ **Important Note**: Independent validation has identified issues that need fixing:
- Phase measurement precision needs recalibration
- Statistical requirements are higher than initially estimated  
- Some dimensional analysis needs correction

See `VALIDATION_STATUS.md` for details and ongoing fixes.

## 🤝 Contributing

We need help with:
- Fixing dimensional analysis in equations
- Improving wave extraction accuracy
- Connecting to LIGO/Virgo data pipelines
- Statistical analysis and parameter estimation

See `CONTRIBUTING.md` for guidelines.

## 📚 References

1. Kaluza, T. (1921). "On the Unification Problem in Physics"
2. Klein, O. (1926). "Quantum Theory and Five-Dimensional Relativity"
3. LIGO Scientific Collaboration - gravitational wave data

## 📜 License

MIT License - see LICENSE file for details.

---

*The hunt for hidden dimensions continues!*
