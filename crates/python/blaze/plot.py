"""Optional band plotting. Install blaze2d[plot] to use this helper."""


def plot(result, *, ax=None):
    """Plot reduced frequencies using the recorded reciprocal-space distances."""
    import matplotlib.pyplot as plt
    if "frequencies" not in result:
        raise ValueError("Select a band result before plotting")
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(result["distances"], result["frequencies"])
    meta = result["metadata"]
    labels, indices = meta.get("labels", []), meta.get("label_indices", [])
    if labels:
        ax.set_xticks(result["distances"][indices], labels)
    ax.set_ylabel("Reduced frequency (c / reference length)")
    ax.set_xlabel("Reciprocal-space path")
    return ax
