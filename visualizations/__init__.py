"""
Visualizations package entrypoint.
Re-exports classes from top-level visualizations.py.
"""
import sys
import os
import importlib.util

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
module_file = os.path.join(root_dir, 'visualizations.py')

if os.path.exists(module_file):
    spec = importlib.util.spec_from_file_location("visualizations_file_module", module_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    ColorPalette = getattr(mod, 'ColorPalette', None)
    TrainingLossVisualizer = getattr(mod, 'TrainingLossVisualizer', None)
    SandwichedBlockVisualizer = getattr(mod, 'SandwichedBlockVisualizer', None)
    DataSliceVisualizer = getattr(mod, 'DataSliceVisualizer', None)
