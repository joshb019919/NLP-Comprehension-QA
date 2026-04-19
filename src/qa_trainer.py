from pathlib import Path

import torch
from datasets import Dataset
from common import release_memory
from evaluation import run_postprocessed_eval
from transformers import Trainer, TrainerCallback


class GradientValueClippingCallback(TrainerCallback):
    def __init__(self, clip_value: float) -> None:
        self._clip_value = float(clip_value)

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if self._clip_value <= 0.0 or model is None:
            return control

        parameters = [param for param in model.parameters() if param.grad is not None]
        if parameters:
            torch.nn.utils.clip_grad_value_(parameters, self._clip_value)
        return control


class BestModelInMemoryCallback(TrainerCallback):
    def __init__(self, metric_name: str, greater_is_better: bool) -> None:
        self._metric_name = metric_name
        self._greater_is_better = bool(greater_is_better)
        self._best_metric: float | None = None
        self._best_step: int | None = None
        self._best_state_path: Path | None = None

    def _is_better(self, candidate: float) -> bool:
        if self._best_metric is None:
            return True
        if self._greater_is_better:
            return candidate > self._best_metric
        return candidate < self._best_metric

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        if metrics is None or model is None or self._metric_name not in metrics:
            return control

        candidate = float(metrics[self._metric_name])
        if not self._is_better(candidate):
            return control

        self._best_metric = candidate
        self._best_step = int(state.global_step)
        best_state_dict = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }
        if self._best_state_path is None:
            self._best_state_path = Path(args.output_dir) / "best-model-snapshot.pt"
        torch.save(best_state_dict, self._best_state_path)
        del best_state_dict
        release_memory()
        return control

    def restore_best_model(self, model) -> tuple[float | None, int | None]:
        if self._best_state_path is None or not self._best_state_path.exists():
            return None, None

        best_state_dict = torch.load(self._best_state_path, map_location="cpu")
        model.load_state_dict(best_state_dict)
        del best_state_dict
        release_memory()
        return self._best_metric, self._best_step


class QATrainer(Trainer):
    def __init__(
        self,
        *args,
        dataset_name: str,
        version_2_with_negative: bool,
        sgd_momentum: float = 0.9,
        raw_eval_examples: Dataset | None = None,
        postprocess_eval_examples: Dataset | None = None,
        postprocess_eval_features: Dataset | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._dataset_name = dataset_name
        self._version_2_with_negative = version_2_with_negative
        self._sgd_momentum = sgd_momentum
        self._raw_eval_examples = raw_eval_examples
        self._postprocess_eval_examples = postprocess_eval_examples
        self._postprocess_eval_features = postprocess_eval_features

    def create_optimizer(self, model=None):
        if self.optimizer is not None:
            return self.optimizer

        if self.args.optim != "sgd":
            return super().create_optimizer(model)

        model = model or self.model

        decay_parameters = self.get_decay_parameter_names(model)
        optimizer_grouped_parameters = [
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if param.requires_grad and name in decay_parameters
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if param.requires_grad and name not in decay_parameters
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = torch.optim.SGD(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            momentum=self._sgd_momentum,
        )
        return self.optimizer

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        should_run_postprocess_eval = (
            metric_key_prefix == "eval"
            and self._raw_eval_examples is not None
            and self._postprocess_eval_examples is not None
            and self._postprocess_eval_features is not None
        )

        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        if should_run_postprocess_eval:
            post_metrics = run_postprocessed_eval(
                trainer=self,
                dataset_name=self._dataset_name,
                version_2_with_negative=self._version_2_with_negative,
                raw_examples=self._raw_eval_examples,
                eval_examples=self._postprocess_eval_examples,
                eval_features_for_postprocess=self._postprocess_eval_features,
                prefix=metric_key_prefix,
            )
            metrics.update(post_metrics)
            added_post_metrics = {
                key: value for key, value in post_metrics.items() if isinstance(value, (int, float))
            }
            if added_post_metrics:
                self.log(added_post_metrics)
                self.control = self.callback_handler.on_evaluate(
                    self.args, self.state, self.control, metrics
                )
        release_memory()

        return metrics
