import matplotlib.pyplot as plt

class ConvergencePlotter:
    def __init__(self):
        self.data = []

    def add_run(self, convergence, label, convergence_index=None, color=None):
        self.data.append({
            "convergence": convergence,
            "label": label,
            "convergence_index": convergence_index,
            "color": color
        })

    def plot(self, title="Convergencia ", xlabel="Iteraciones", ylabel="Valor Total", save_path=None):
        plt.figure(figsize=(10, 6))
        for entry in self.data:
            convergence = entry["convergence"]
            label = entry["label"]
            idx = entry["convergence_index"]
            color = entry["color"]

            plt.plot(convergence, label=label, linewidth=2, color=color)

            if idx is not None and idx < len(convergence):
                plt.axvline(x=idx, linestyle='--', color=color or 'gray', alpha=0.5)

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()

        if save_path:
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
        else:
         plt.show()
