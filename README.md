# XY_Simulation

Oblique incidence hopefully

To run this notebook, first install requirements.txt in an evironment using:

```
pip install -r requirements.txt
```

Now, you can run Run_Simulation.ipynb as a jupyter notebook.

You can change the resolution and number of simulations directly in the notebook.

In thermalNoiseParams.yaml, you can change how many times you want to decross images, how many types of masks you want to add, simulation iterations (steps), number of snapshots. Also, decross angle, number of defects, defect clustering, temperature, step size and something.

In defectData.yaml, 116-120 allow you to change from normal to oblique incidence, and the oblique() function has a lot of physical parameters that can be changed.

Finally, there are a lot of noise mask parameters in maskTemplate.yaml.
