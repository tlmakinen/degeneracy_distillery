# Rosenbrock NPE scaling figure

This folder redraws `rosenbrock_npe_scaling.png`. The points are means
over trials with `status=ok`. The error bars are the standard error of
those trials. `d=32` uses the 8 trials that produced a posterior score.

```bash
python plot_rosenbrock_npe_scaling.py
```

`metrics.csv` is the input. Discovered is stored already shifted onto the
oracle-axis scale. `rosenbrock_scaling_brief.md` describes the screen,
the discovery steps, and the NPE.
