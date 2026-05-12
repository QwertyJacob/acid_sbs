import wandb
import plotly.express as px
import torch
from sklearn.decomposition import PCA
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Reporter:
    def __init__(self, args: dict):
        w = args["wandb"]
        self.wbt = w["wb_tracking"]
        self.wb_run = wandb.init(
            project=w["wb_project_name"],
            name=w["wb_run_name"],
            group=w["wb_group_name"],
            config=args,
            mode="online" if self.wbt else "disabled",
        )
        logger.info(
            f"WandB {'enabled' if self.wbt else 'disabled'} | "
            f"project={w['wb_project_name']} run={w['wb_run_name']}"
        )

    def log_scalars(self, metrics: dict, step: int):
        if self.wbt and self.wb_run is not None:
            self.wb_run.log(metrics, step=step)

    def plot_hidden_space(
        self,
        hiddens: torch.Tensor,
        labels: torch.Tensor,
        class_names: list,
        phase: str,
        step: int,
    ):
        """
        Projects hidden vectors to 2D with PCA (if dim > 2) and logs a Plotly
        scatter plot to WandB, coloured by class label.
        """
        if not self.wbt or self.wb_run is None:
            return

        h_np = hiddens.detach().cpu().numpy()
        l_np = labels.detach().cpu().numpy()

        if h_np.shape[1] > 2:
            pca = PCA(n_components=2, random_state=0)
            h_np = pca.fit_transform(h_np)

        nl_labels = [class_names[i] if i < len(class_names) else f"class_{i}" for i in l_np]
        data_dict = {
            "PC_1": h_np[:, 0],
            "PC_2": h_np[:, 1],
            "class": nl_labels,
        }
        fig = px.scatter(
            data_dict, x="PC_1", y="PC_2", color="class",
            title=f"{phase} Hidden Space (PCA)",
        )
        fig.update_traces(marker=dict(size=7, opacity=0.7))
        self.wb_run.log({f"{phase}/hidden_space": fig}, step=step)
        logger.debug(f"Logged {phase} hidden space plot at step {step}.")

    def finish(self):
        if self.wb_run is not None:
            self.wb_run.finish()
        logger.info("WandB run finished.")
