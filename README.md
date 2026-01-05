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

## Repository Structure

Typical components:

- **Base class**
  - Defines the common interface for all topology species
- **Concrete topology species**
  - Specific geometry families (e.g. parametric shapes)
- **Free-form species**
  - Random or parameter-driven topology generation
- **Utility functions**
  - Discretization, masks, or geometry helpers

(See source files for implementation details.)

---

## Example Usage

```python
from topology_species import Bezier

topology = Bezier()
topology.draw_shape()
