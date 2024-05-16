import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from robust.plot import setup_plot

c = setup_plot()
c = sns.color_palette("colorblind")

for i, ci in enumerate(c.as_hex()):
    print(f"\definecolor{{color{i}}}{{HTML}}{{{ci[1:]}}}")


def tukey(r, c):
    if type(r) is np.ndarray:
        out = np.zeros_like(r)
        in_basin = np.abs(r) <= c
        out[~in_basin] = c**2 / 6
        out[in_basin] = (c**2 / 6) * (1 - (1 - (r[in_basin] / c) ** 2) ** 3)
        return out
    else:
        if np.abs(r) <= c:
            return (c**2 / 6) * (1 - (1 - (r / c) ** 2) ** 3)
        else:
            return c**2 / 6


fig, ax = plt.subplots(1, 1, figsize=(3.5, 1.75), layout="constrained", dpi=300)
c = 1.0
xlim = [-(c**2) / 6 - 2 * c, c**2 / 6 + 2 * c]
ylim = [-0.05 * (c**2) / 6, 1.1 * c**2 / 6]
x = np.linspace(xlim[0], xlim[1], 100)
y = tukey(x, c)

sns.lineplot(x=x, y=y, ax=ax)
ax.set_xlim(xlim)
ax.set_ylim(ylim)

plt.savefig("figures/tukey.pdf", dpi=300)
