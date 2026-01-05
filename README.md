# Topology Species for Meta-Atom Design

This repository provides a collection of **2D topology “species”** for meta-atom design.  
Each topology is implemented as a Python class following a common interface, enabling
easy reuse, extension, and integration into simulation or inverse-design pipelines.

The goal is to standardize how different topology families (parametric or free-form)
are generated while keeping them interchangeable.

---

## Core Idea

- Each **topology species** is a Python class
- All species share a **common base interface**
- New topologies can be added via **inheritance**
- Supports:
  - Parametric geometries
  - Randomly generated free-form topologies
  - Programmatic topology generation from parameters

This makes it easy to:
- benchmark different topology families
- swap geometries in optimization loops
- use the same downstream simulation code

---

## 📁 Repository Structure

This repository is intentionally minimal:

| File | Description |
|------|-------------|
| `topology_species.py` | Core implementation of all topology species and the shared base interface |
| `example.py` | Example script that instantiates and visualizes each topology species once |

There are no external dependencies beyond standard scientific Python packages.

---

## 🧬 Available Topology Species

The following topology species are currently implemented.  
All species share a common interface and can be used interchangeably.

| Name | Category | Source |
|-----|----------|--------|
| `Cross` | Basic shape | - |
| `Rectangle` | Basic shape | - |
| `Ellipse` | Basic shape | - |
| `SplitRing` | Basic shape | - |
| `VShape` | Basic shape | - |
| `LShape` | Basic shape | - |
| `Bezier` | Freeform (parametric) | This work |
| `BezierFlower` | Freeform (parametric) | This work |
| `BezierStar` | Freeform (parametric) | This work |
| `NeedleDrop` | Freeform (stochastic) | Opt. Express 28, 31932-31942 (2020) |
| `HeightmapSlice` | Freeform (procedural) | Opt. Express 28, 24229-24242 (2020) |
| `DiffusionAggregation` | Freeform (procedural) | This work |
| `WaveInterference` | Freeform (procedural) | This work |
| `CellularAutomata` | Freeform (procedural) | This work |

**Categories**
- **Basic shape**: simple analytic geometries with few parameters  
- **Freeform**: procedurally or parametrically generated topologies, often higher complexity

---

## Example Usage

```python
from topology_species import Bezier

topology = Bezier()
topology.draw_shape()
