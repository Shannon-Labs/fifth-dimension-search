# Known Issues & Bugs

> **TL;DR**: This code is broken in fundamental ways. We're sharing it anyway to get community help fixing it.

## 🔴 CRITICAL - Physics Fundamentally Wrong

### 1. No Real 5D Physics Implemented
- **Problem**: We claim to simulate 5D gravity but only evolve 3D spacetime
- **Location**: `src/fifth_dimension_search/brane_world.py:249` - Laplacian only sums over 3D
- **Evidence**: `laplacian = d2_dx2 + d2_dy2 + d2_dz2` (no d2_dW2 term!)
- **Impact**: This is not extra-dimensional physics in any sense
- **Reality Check**: We're fooling ourselves about what we're computing

### 2. Strain Scaling Wrong by 10^10 Factor
- **Problem**: (Resolved in current branch) geometric-to-observer conversion under-estimated strains by ~10⁹
- **Location**: `src/fifth_dimension_search/template_bank.py:88-95`, wave extraction in `src/fifth_dimension_search/brane_world.py`
- **Evidence**: `datasets/phase_amp_summary.csv` now reports peak_h_plus ≈ 6.4×10⁻²¹ after the fix. Previous revisions produced 5×10⁻³⁰.
- **Impact**: Physical plausibility restored for the toy data, although the underlying physics model is still inconsistent
- **Reality Check**: Further validation against calibrated waveforms is required before trusting any quantitative claims

### 3. Not Actually KK Theory
- **Problem**: Claims "Kaluza-Klein gravity" but implements random scalar field + BSSN
- **Location**: Throughout `src/fifth_dimension_search/brane_world.py`
- **Evidence**: No 5D→4D dimensional reduction, no Einstein-Maxwell-Dilaton coupling
- **Impact**: Results have no connection to real extra-dimensional physics
- **Reality Check**: This isn't any known theoretical framework

## 🟡 MAJOR - Theory Inconsistent & Numerics Questionable

### 4. Theory Framework Inconsistent
- **Problem**: Mixing concepts from Kaluza-Klein, brane-world, and scalar-tensor theories
- **Location**: Throughout codebase documentation and variable names
- **Evidence**: Claims KK theory but uses brane-world matter confinement assumptions
- **Impact**: No coherent theoretical prediction to test against
- **What's Needed**: Choose one consistent framework and implement it properly

### 5. No Convergence Testing
- **Problem**: Never verified that results converge with higher resolution
- **Location**: All simulation outputs
- **Impact**: Could be measuring numerical errors, not physics
- **What's Needed**: Run same configurations at multiple resolutions

### 6. Unphysical Boundary Conditions
- **Problem**: Artificial boundary reflections contaminate gravitational waves
- **Location**: `src/fifth_dimension_search/brane_world.py:706-713` - only periodic on one axis
- **Code**: `tensor[...,0] = tensor[...,-1]` (other axes create reflections)
- **Impact**: "Extra-dimensional effects" might just be boundary artifacts
- **What's Needed**: Proper absorbing boundaries or full periodicity

## 🟢 ACKNOWLEDGED - Overestimated Capabilities & Sensitivity

### 7. Overestimated LIGO Phase Sensitivity
- **Our Original Claim**: LIGO can measure phases to 10^-6 radian precision
- **Physical Reality**: Phase uncertainty ~ 0.1 rad even for loudest events
- **Impact**: Even if our physics were right, effects would be undetectable
- **Status**: ✅ Now honestly documented

### 8. Statistical Requirements Underestimated by 100x
- **Our Original Claim**: 100 merger events sufficient for 5σ discovery
- **Physical Reality**: Would need 10,000+ events (if effects existed and were detectable)
- **Impact**: Not feasible with current or near-future gravitational wave catalogs
- **Status**: ✅ Now realistic about requirements

### 9. Sensitivity Claims Wildly Optimistic
- **Our Approach**: Assumed toy model effects would be easily detectable
- **Physical Reality**: Real extra-dimensional effects (if any) would be much subtler
- **Impact**: Built unrealistic expectations about what's possible to measure
- **Status**: ✅ Now acknowledge this was wishful thinking

## 🔧 Priority Fix Order

### What to Tackle First:
1. **Fix strain amplitude scaling** - Currently wrong by factor of 10^10, makes everything meaningless
2. **Implement actual 5D physics OR admit it's a 3D toy model** - Stop claiming 5D when we only evolve 3D
3. **Choose one consistent theory framework** - KK, brane-world, or scalar-tensor (not a confused mix)
4. **Add dimensional analysis checking** - Make sure equations are mathematically valid
5. **Implement convergence testing** - Verify we're measuring physics, not numerical errors

### Secondary Issues:
- Fix boundary conditions to eliminate reflections
- Implement proper wave extraction
- Add uncertainty quantification
- Clean up unit conversions

## How to Help Fix This

**Everyone Can Contribute**:
- **Confirm our bug reports** - Run the code and verify you see the same issues
- **Suggest better approaches** - Know how extra-dimensional physics should work?
- **Implement fixes** - Pick any issue above and work on it
- **Ask questions** - Help us understand what we got wrong

**How to Contribute**:
1. Open a GitHub issue about which problem interests you
2. Fork the repo and experiment with fixes
3. Submit PRs with clear before/after comparisons
4. Help us learn from the physics and numerical relativity communities

**We Want to Learn**: This project failed in its original goals, but we can make it succeed as a collaborative learning experience.

---

*"The first principle is that you must not fool yourself — and you are the easiest person to fool."* - Richard Feynman

We definitely fooled ourselves. **Help us turn this broken attempt into something that might actually work!**