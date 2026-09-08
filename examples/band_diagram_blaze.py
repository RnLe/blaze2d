"""Plot TE and TM studies with the shared configuration and dictionary API."""
from pathlib import Path
import matplotlib.pyplot as plt
import blaze

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, name in zip(axes, ['square-rods', 'triangular-holes']):
    config = blaze.Config.from_file(Path(__file__).parent / 'calculations' / f'{name}.toml').to_dict()
    config['sweeps'] = [{'name':'polarization','target':'polarization','values':['TM','TE']}]
    study = blaze.run(config, threads=2)
    if study['errors']: raise RuntimeError(study['errors'])
    for result, color in zip(study['results'], ['C0','C1']):
        ax.plot(result['distances'], result['frequencies'], color=color, linewidth=.8)
        ax.plot([], [], color=color, label=result['metadata']['config']['polarization'])
    first = study['results'][0]
    ax.set_xticks(first['distances'][first['metadata']['label_indices']], first['metadata']['labels'])
    ax.set_title(name.replace('-', ' ').title())
    ax.set_ylabel('Reduced frequency (c/reference length)')
    ax.legend()
fig.tight_layout()
fig.savefig('band_diagram_blaze.svg')
plt.show()
