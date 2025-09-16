# 🧪 Fifth Dimension Research Sandbox: Exploring Extra Dimensions with Gravitational Waves

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CUDA/MPS Support](https://img.shields.io/badge/accelerated-CUDA%2FMPS-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **⚠️ RESEARCH SANDBOX DISCLAIMER: This is exploratory code with known physics issues, NOT a validated detection pipeline**

## What This IS

- 🧪 **Research sandbox** for exploring extra-dimensional gravity effects
- 🤝 **Community collaboration project** seeking physics input and code contributions
- 📚 **Educational exploration** of how extra dimensions might affect gravitational waves
- 🔬 **Toy model** for testing numerical relativity techniques
- 💡 **Starting point** for future rigorous implementations

## What This IS NOT

- ❌ **NOT a validated detection pipeline** - contains fundamental physics errors
- ❌ **NOT real Kaluza-Klein theory** - simplified toy model at best
- ❌ **NOT ready for scientific publication** - needs extensive physics fixes
- ❌ **NOT claiming any detections** - results are exploratory only
- ❌ **NOT suitable for actual LIGO/Virgo analysis** without major corrections

## 🚨 Critical Issues We Know About

This code has **fundamental problems** we're actively working to fix:
- Strain amplitudes wrong by factor of 10^10 (completely unphysical)
- No actual 5D physics implemented (claims 5D but only evolves 3D)
- Dimensional analysis errors throughout
- Not implementing any consistent theory

See `KNOWN_ISSUES.md` for complete list. **We share this broken code to get community help fixing it!**

## 🎯 Our Ambitious but Broken Attempt

We're exploring whether gravitational waves could reveal extra dimensions by comparing black hole vs neutron star mergers. The physics intuition is sound, but our implementation has serious bugs we need help fixing.

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

## 📊 The Physics Concept (What We're Trying to Model)

In 1921, Theodor Kaluza showed that Einstein's General Relativity in 5 dimensions naturally unifies gravity and electromagnetism. Modern theories suggest extra dimensions could be detectable through gravitational wave observations.

### Key Parameters (In Our Toy Model):
- `q`: KK charge - coupling to extra dimension (10⁻⁴ to 10⁻³)
- `L₅`: Compactification radius - size of hidden dimension
- `m₅`: 5D mass parameter - energy scale of effects

**⚠️ Important**: Our implementation of these concepts is fundamentally flawed - see issues below.

## 🎯 Project Status: Broken but Collaborative

**Current Reality**: This code has serious physics problems that make results meaningless:
- No actual 5D evolution implemented despite claims
- Strain scaling wrong by 10 orders of magnitude
- Dimensional analysis errors throughout equations
- Not implementing any consistent extra-dimensional theory

**Our Approach**: Share the broken code openly and invite community collaboration to fix it.

**Help Wanted**: Theoretical physicists, numerical relativity experts, and anyone interested in making this concept work properly!

## 🤝 How You Can Help

**Priority Issues** (pick any that interest you):
1. **Fix strain scaling** - figure out why amplitudes are 10^10 too small
2. **Implement real 5D physics** - currently we only evolve 3D despite claiming 5D
3. **Fix dimensional analysis** - equations mix dimensionless and dimensional quantities
4. **Choose consistent theory** - decide between KK, brane-world, or honest toy model
5. **Add convergence testing** - verify numerical results don't depend on resolution

**How to Contribute**:
- Open an issue discussing which problem you want to tackle
- Fork the repo and work on fixes with clear documentation
- Submit PRs with before/after comparisons showing improvements
- Help us understand what real extra-dimensional theories predict

**All Skill Levels Welcome**: Even just confirming our bug reports is valuable!

## 📚 References

1. Kaluza, T. (1921). "On the Unification Problem in Physics"
2. Klein, O. (1926). "Quantum Theory and Five-Dimensional Relativity"
3. LIGO Scientific Collaboration - gravitational wave data

## 📜 License

MIT License - see LICENSE file for details.

---

*"Here's our ambitious but broken attempt at detecting extra dimensions with gravitational waves - help us make it real!"*

**Join the collaboration**: Together we can turn this broken toy model into something that might actually work.
