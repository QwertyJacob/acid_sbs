import threading
import logging
import torch
import torch.nn.functional as F

from src.model import MLPEncoder, PrototypicalClassifier
from src.data import init_class_centers, sample_episode, sample_test_batch, get_class_names
from src.reporter import Reporter
from src.utils import set_seed

logger = logging.getLogger(__name__)


class Experiment:
    def __init__(self, args: dict):
        self.args = args
        self._stop_event = threading.Event()
        set_seed(args["experiment"]["seed"])
        self.device = args["experiment"]["device"]

        # Fix class centers in memory (tiny: [n_classes, n_dims])
        self.centers = init_class_centers(args)
        self.class_names = get_class_names(args)

        # Model
        pc = args["prototypical"]
        self.encoder = MLPEncoder(
            in_dim=args["data"]["n_dims"],
            hidden_dim=pc["hidden_dim"],
            n_layers=pc["n_layers"],
            dropout=pc["dropout"],
        ).to(self.device)
        self.classifier = PrototypicalClassifier()
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=pc["learning_rate"])

        # Reporter (WandB)
        self.reporter = Reporter(args)
        self.step = 0

        logger.info(
            f"Experiment ready | device={self.device} | "
            f"n_dims={args['data']['n_dims']} | "
            f"train_classes={args['data']['n_train_classes']}/{args['data']['n_classes']} | "
            f"hidden_dim={pc['hidden_dim']} | episodes={pc['train_episodes']}"
        )

    def stop(self):
        logger.info("Stop requested.")
        self._stop_event.set()

    def run(self):
        pc = self.args["prototypical"]
        report_every = pc["report_every"]
        total = pc["train_episodes"]
        logger.info(f"Training started — {total} episodes, reporting every {report_every}.")

        for episode in range(1, total + 1):
            if self._stop_event.is_set():
                logger.info("Stop event detected. Exiting training loop.")
                break

            loss, acc, train_hiddens, train_labels = self._train_episode()
            self.step += 1

            if episode % report_every == 0:
                test_metrics = self._test()
                logger.info(
                    f"[ep {episode:>4}/{total}] "
                    f"train_loss={loss:.4f}  train_acc={acc:.3f}  "
                    f"test_acc={test_metrics['accuracy']:.3f}"
                )
                self.reporter.log_scalars(
                    {
                        "train/loss": loss,
                        "train/accuracy": acc,
                        "test/accuracy": test_metrics["accuracy"],
                    },
                    step=self.step,
                )
                if self.args["wandb"]["plots"]:
                    self.reporter.plot_hidden_space(
                        hiddens=train_hiddens,
                        labels=train_labels,
                        class_names=self.class_names,
                        phase="Train",
                        step=self.step,
                    )
                    self.reporter.plot_hidden_space(
                        hiddens=test_metrics["hiddens"],
                        labels=test_metrics["labels"],
                        class_names=self.class_names,
                        phase="Test",
                        step=self.step,
                    )

        logger.info("Training loop finished. Finalising reporter.")
        self.reporter.finish()

    def _train_episode(self):
        """One prototypical episode — data drawn fresh from Gaussian centers."""
        self.encoder.train()
        support_x, support_y, query_x, query_y = sample_episode(
            self.centers, self.args, self.device
        )

        support_h = self.encoder(support_x)
        query_h = self.encoder(query_x)
        logits, _ = self.classifier(support_h, support_y, query_h)

        loss = F.cross_entropy(logits, query_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        preds = logits.argmax(dim=1)
        acc = (preds == query_y).float().mean().item()
        return loss.item(), acc, query_h.detach().cpu(), query_y.cpu()

    @torch.no_grad()
    def _test(self):
        """
        Draws a fresh test batch covering ALL classes (known + unknown).
        Prototypes are built from a fresh known-class support draw.
        Unknown-class samples will mostly be misclassified — correct and expected
        behaviour for a prototypical net without unseen support.
        """
        self.encoder.eval()
        support_x, support_y, test_x, test_y = sample_test_batch(
            self.centers, self.args, self.device
        )

        support_h = self.encoder(support_x)
        test_h = self.encoder(test_x)
        logits, _ = self.classifier(support_h, support_y, test_h)

        preds = logits.argmax(dim=1)
        acc = (preds == test_y).float().mean().item()
        return {
            "accuracy": acc,
            "hiddens": test_h.cpu(),
            "labels": test_y.cpu(),
        }
