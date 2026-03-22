import time
import numpy as np
import matplotlib.pyplot as plt
from generator import wind, thunder, rain, ocean, leaves, buzzing_electronics, grumbling_machinery, faraway_cars, crickets

sounds = {
    "wind": wind,
    "thunder": thunder,
    "rain": rain,
    "ocean": ocean,
    "leaves": leaves,
    "buzzing_electronics": buzzing_electronics,
    "grumbling_machinery": grumbling_machinery,
    "faraway_cars": faraway_cars,
    "crickets": crickets,
}

times = {}
for name, fn in sounds.items():
    runs = []
    for _ in range(3):
        start = time.perf_counter()
        fn()
        runs.append((time.perf_counter() - start) * 1000)
    times[name] = np.mean(runs)

labels = list(times.keys())
values = list(times.values())

fig, ax = plt.subplots(figsize=(6, 6))
bars = ax.bar(labels, values, width=0.4, color="steelblue", edgecolor="black")
ax.set_xlabel("Sound")
ax.set_ylabel("Time (ms)")
ax.set_title("Average Render Time per Sound (3 runs)")
plt.xticks(rotation=45, ha="right")

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{val:.1f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("analysis.png", dpi=150)
plt.show()
