#!/usr/bin/env python
# coding: utf-8

"""
Script to analyze the coordination number (CN) of cations with water molecules 
from AIMD trajectory files (XDATCAR format). The script uses ASE to read 
trajectories, constructs neighbor lists to identify cation-O and O-H bonds, 
and calculates average CN and cation-surface distances over selected timesteps.
"""

from ase.io import read
import numpy as np
from ase.neighborlist import NeighborList, natural_cutoffs


def get_cation_CN(atoms, cation):
    """
    Compute coordination number (CN) and z-distance of cations in a given structure.

    Parameters
    ----------
    atoms : ase.Atoms
        ASE Atoms object representing the system.
    cation : str
        Symbol of the cation to analyze (e.g., 'K', 'Cs', 'Li').

    Returns
    -------
    x_array : list of int
        Coordination numbers (CN) of each cation with water molecules.
    z_array : list of float
        z-distance of each cation from the reference metal surface.
    """

    # Parameters for neighbor list
    radii_multiplier = 1.1
    skin = 0.25

    # Get indices of all cations
    cation_indices = [atom.index for atom in atoms if atom.symbol == cation]

    # Reference metal surface index (assume first atom type is the metal)
    metal_indices = [atom.index for atom in atoms if atom.symbol == atoms[0].symbol]

    # Build neighbor list for identifying cation-oxygen neighbors
    nl = NeighborList(
        natural_cutoffs(atoms, radii_multiplier),
        self_interaction=False,
        bothways=True,
        skin=skin,
    )
    nl.update(atoms)

    x_array = []  # coordination numbers
    z_array = []  # z-distances

    for c_index in cation_indices:
        # z-distance relative to bottom-most metal atom
        z_dist = atoms.get_positions()[c_index][2] - atoms.get_positions()[metal_indices[-1]][2]

        # Find neighboring oxygen atoms around the cation
        indices, _ = nl.get_neighbors(c_index)
        oxygen_indices = [j for j in indices if atoms[j].symbol == "O"]

        # Distances from cation to neighboring oxygen atoms
        distances = atoms.get_distances(c_index, oxygen_indices, mic=True)

        # Build a tighter neighbor list for oxygen-hydrogen bonding
        nl_h = NeighborList(
            natural_cutoffs(atoms, 0.7),
            self_interaction=False,
            bothways=True,
            skin=skin,
        )
        nl_h.update(atoms)

        # Check which oxygens are part of intact water molecules
        water_oxygens = []
        water_distances = []
        for k, o_index in enumerate(oxygen_indices):
            neigh, _ = nl_h.get_neighbors(o_index)
            if any(atoms[j].symbol == "H" for j in neigh):
                water_oxygens.append(o_index)
                water_distances.append(distances[k])

        # Coordination number = number of water molecules bonded
        x_array.append(len(water_oxygens))
        z_array.append(z_dist)

    return x_array, z_array


# -------------------------------------------------------------------------
# Load trajectory
# -------------------------------------------------------------------------
try:
    atoms = read("XDATCAR", index=":")
except:
    atoms = read("XDATCAR.bz2", index=":")

steps = len(atoms)
print(f"Number of trajectory steps: {steps}")


# -------------------------------------------------------------------------
# Identify the cation type (exclude metal, C, H, O)
# -------------------------------------------------------------------------
symbols = atoms[0].get_chemical_symbols()
cation = None
for symbol in symbols:
    if symbol not in [atoms[0][0].symbol, "C", "H", "O"]:
        cation = symbol
        break

print(f"Cation identified: {cation}")


# -------------------------------------------------------------------------
# Calculate CN and z-distance over last 3000 steps (sampled every 25)
# -------------------------------------------------------------------------
coord_array = []
z_dist_array = []

for i in range(steps - 3000, steps, 25):
    coord, z_dist = get_cation_CN(atoms[i], cation)
    coord_array.append(coord)
    z_dist_array.append(z_dist)

# Final frame analysis
atoms_final = atoms[-1]
coord_final, z_dist_final = get_cation_CN(atoms_final, cation)


# -------------------------------------------------------------------------
# Post-processing: calculate average CN and z-distance for each cation
# -------------------------------------------------------------------------
for j in range(len(coord_array[0])):
    coord_single = [coord_array[p][j] for p in range(len(coord_array))]
    z_single = [z_dist_array[p][j] for p in range(len(z_dist_array))]

    print(f"\nAnalysis for cation {j+1}:")
    print(f"  Average z-distance: {np.average(z_single):.2f} ± {np.std(z_single):.2f} Å")
    print(f"  Average coordination number: {np.average(coord_single):.1f} ± {np.std(coord_single):.2f}")
