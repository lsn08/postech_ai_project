from __future__ import annotations

import numpy as np
import tensorflow as tf
from dataclasses import dataclass, asdict
from typing import Optional, Union, Tuple, Dict, Any
import tarfile
import pickle
import csv
from keras.utils import to_categorical
from keras import layers, models
from typing import List
from pathlib import Path
from typing import Literal


class CIFAR100Loader:

    def __init__(
            self,
            tar_path: str,
            extract_dir: str = "./cifar100_extracted",
            label_mode: Literal["fine", "coarse"] = "fine",
            normalize: bool = True,
            one_hot: bool = True,
            channels_last: bool = True,
            force_extract: bool = False,
    ) -> None:
        self.tar_path = Path(tar_path)
        self.extract_dir = Path(extract_dir)
        self.dataset_root = self.extract_dir / "cifar-100-python"
        self.label_mode = label_mode
        self.normalize = normalize
        self.one_hot = one_hot
        self.channels_last = channels_last
        self.force_extract = force_extract

        self._meta: Optional[Dict[bytes, List[bytes]]] = None
        self._num_classes: int = 100 if label_mode == "fine" else 20
        self._label_key: bytes = b"fine_labels" if label_mode == "fine" else b"coarse_labels"

    def load(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        self._maybe_extract()
        self._load_meta()
        x_train, y_train = self._load_split("train")
        x_test, y_test = self._load_split("test")
        return (x_train, y_train), (x_test, y_test)

    def get_label_names(self) -> List[str]:
        if self._meta is None:
            self._load_meta()
        key = b"fine_label_names" if self.label_mode == "fine" else b"coarse_label_names"
        return [n.decode("utf-8") for n in self._meta[key]]

    def fine_label_names(self) -> List[str]:
        if self._meta is None:
            self._load_meta()
        return [n.decode("utf-8") for n in self._meta[b"fine_label_names"]]

    def coarse_label_names(self) -> List[str]:
        if self._meta is None:
            self._load_meta()
        return [n.decode("utf-8") for n in self._meta[b"coarse_label_names"]]

    def class_counts(self, y: np.ndarray) -> np.ndarray:
        if y.ndim == 2:
            y_idx = np.argmax(y, axis=-1)
        else:
            y_idx = y
        return np.bincount(y_idx, minlength=self._num_classes)

    def _maybe_extract(self) -> None:
        if not self.dataset_root.exists() or self.force_extract:
            self.extract_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(self.tar_path, "r:gz") as tar:
                tar.extractall(path=self.extract_dir)

    def _load_meta(self) -> None:
        meta_path = self.dataset_root / "meta"
        self._meta = self._unpickle(meta_path)

    def _load_split(self, split: Literal["train", "test"]) -> Tuple[np.ndarray, np.ndarray]:
        path = self.dataset_root / split
        d = self._unpickle(path)
        x = d[b"data"]
        y = np.array(d[self._label_key], dtype=np.int32)

        x = x.reshape(-1, 3, 32, 32)
        if self.channels_last:
            x = x.transpose(0, 2, 3, 1)
        x = x.astype(np.float32)
        if self.normalize:
            x /= 255.0

        if self.one_hot:
            y = to_categorical(y, num_classes=self._num_classes).astype(np.float32)
        return x, y

    @staticmethod
    def _unpickle(file_path: Path) -> Dict[bytes, np.ndarray]:
        with open(file_path, "rb") as fo:
            d = pickle.load(fo, encoding="bytes")
        return d


# 사용 예시
if __name__ == "__main__":
    loader = CIFAR100Loader(
        tar_path="cifar-100-python.tar.gz",
        label_mode="fine",
        normalize=True,
        one_hot=True,
        channels_last=True,
        force_extract=False,
    )

    (x_train, y_train), (x_test, y_test) = loader.load()
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    print("#classes:", y_train.shape[-1])
    print("per-class counts (train):", loader.class_counts(y_train)[:10], "...")
    print("first 10 fine label names:", loader.fine_label_names()[:10])


# ---------------------------------------------------------------
# 손실 팩토리
# ---------------------------------------------------------------
class LossFactory:
    def __init__(
            self,
            prev_probs_var: Optional[tf.Variable] = None,
            alpha: float = 0.5,
            beta: float = 0.2,
            gamma: float = 0.1,
            delta: float = 0.15,
            epsilon: float = 0.05,
            tau: float = 0.9,
            focal_gamma: float = 2.0,
            focal_alpha: float = 0.25,
            label_smoothing: float = 0.1,
            csv_log_path: Optional[str] = None,
            add_header: bool = True,
    ) -> None:
        self.prev_probs_var = prev_probs_var
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.tau = tau
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.label_smoothing = label_smoothing

        # csv logging
        self._csv_path: Optional[Path] = Path(csv_log_path) if csv_log_path else None
        self._add_header = add_header
        self._step: int = 0  # loss_fn 호출 카운터 (epoch이 아닌 step 기준)

        # base losses
        self.ce = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=label_smoothing,
            reduction=tf.keras.losses.Reduction.NONE,
        )
        self.kld = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)

        if self._csv_path:
            self._init_csv()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def build(self) -> tf.keras.losses.Loss:
        """Return a tf.keras.losses.Loss-like callable."""
        # capture locals to avoid attribute lookups inside tf.function
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma
        delta = self.delta
        tau = self.tau
        focal_gamma = self.focal_gamma
        focal_alpha = self.focal_alpha
        ce = self.ce
        kld = self.kld
        prev_probs_var = self.prev_probs_var

        def loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            # 1) CE
            ce_loss = ce(y_true, y_pred)  # (B,)

            # 2) KL (Consistency with prev_probs)
            if prev_probs_var is None:
                kl_loss = tf.zeros_like(ce_loss)
            else:
                batch = tf.shape(y_pred)[0]
                # prev_probs_var 길이가 배치보다 짧을 수 있으므로 안전하게 slice
                max_len = tf.shape(prev_probs_var)[0]
                batch = tf.minimum(batch, max_len)
                prev_probs_batch = prev_probs_var[:batch]
                y_pred_batch = y_pred[:batch]
                kl_loss = kld(prev_probs_batch, y_pred_batch)

            # 3) Entropy Regularization
            y_pred_c = tf.clip_by_value(y_pred, 1e-7, 1.0)
            entropy = -tf.reduce_sum(y_pred_c * tf.math.log(y_pred_c), axis=-1)

            # 4) Pseudo-label Masked CE
            probs = tf.reduce_max(y_pred, axis=-1)
            mask = tf.cast(probs >= tau, tf.float32)
            pseudo_loss = tf.reduce_sum(ce_loss * mask) / (tf.reduce_sum(mask) + 1e-7)

            # 5) Focal Loss
            ce_focal = -y_true * tf.math.log(y_pred_c)
            pt = tf.reduce_sum(y_true * y_pred_c, axis=-1)
            mod = tf.pow(1.0 - pt, focal_gamma)
            focal_weight = focal_alpha * mod
            focal_loss = tf.reduce_mean(focal_weight * tf.reduce_sum(ce_focal, axis=-1))

            # 6) Total Loss (나머지 가중치는 focal에 할당)
            rest = tf.maximum(1.0 - (alpha + beta + gamma + delta), 0.0)
            total = (
                    alpha * tf.reduce_mean(ce_loss)
                    + beta * tf.reduce_mean(kl_loss)
                    + gamma * tf.reduce_mean(entropy)
                    + delta * pseudo_loss
                    + rest * focal_loss
            )

            # CSV logging (step 단위)
            if self._csv_path:
                self._step += 1
                self._write_csv(
                    step=self._step,
                    ce=float(tf.reduce_mean(ce_loss).numpy()),
                    kl=float(tf.reduce_mean(kl_loss).numpy()),
                    entropy=float(tf.reduce_mean(entropy).numpy()),
                    pseudo=float(pseudo_loss.numpy()),
                    focal=float(focal_loss.numpy()),
                    total=float(total.numpy()),
                )

            return total

        return loss_fn

    # ------------------------------------------------------------------
    # CSV helpers
    # ------------------------------------------------------------------
    def _init_csv(self) -> None:
        assert self._csv_path is not None
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        if self._csv_path.exists() and not self._add_header:
            return
        with open(self._csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "ce", "kl", "entropy", "pseudo", "focal", "total"])  # header

    def _write_csv(self, *, step: int, ce: float, kl: float, entropy: float, pseudo: float, focal: float,
                   total: float) -> None:
        assert self._csv_path is not None
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([step, ce, kl, entropy, pseudo, focal, total])


# ---------------------------------------------------------------
# EMA Buffer
# ---------------------------------------------------------------
class EMAPredBuffer:
    def __init__(self, decay: float = 0.95):
        if not (0.0 < decay < 1.0):
            raise ValueError("decay must be in (0,1)")
        self.decay = decay
        self._value: Optional[tf.Variable] = None

    def update(self, probs: np.ndarray | tf.Tensor) -> tf.Variable:
        probs = tf.convert_to_tensor(probs, dtype=tf.float32)
        if self._value is None:
            self._value = tf.Variable(probs, trainable=False)
        else:
            self._value.assign(self.decay * self._value + (1.0 - self.decay) * probs)
        return self._value

    @property
    def value(self) -> Optional[tf.Variable]:
        return self._value

    def reset(self) -> None:
        self._value = None


# ---------------------------------------------------------------
# Recycling Trainer (AlphaFold 스타일)
# ---------------------------------------------------------------
class RecyclingTrainer:
    def __init__(
            self,
            model: tf.keras.Model,
            x_train: Union[np.ndarray, tf.data.Dataset],
            y_train: Union[np.ndarray, tf.data.Dataset],
            x_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            cfg: TrainConfig = TrainConfig(),
    ) -> None:
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.cfg = cfg

        self.ema_buf = EMAPredBuffer(decay=cfg.ema_decay)
        self.loss_factory: Optional[LossFactory] = None
        self.history_per_cycle: list[Dict[str, Any]] = []

        # 초기 prev_probs를 균등 분포로 셋업 (numpy array 학습일 때만 안전하게 가능)
        if isinstance(self.y_train, np.ndarray):
            n, c = self.y_train.shape
            init = np.full((n, c), 1.0 / c, dtype=np.float32)
            self.ema_buf.update(init)

        self._compile_with_current_loss()

    def _compile_with_current_loss(self) -> None:
        self.loss_factory = LossFactory(
            prev_probs_var=self.ema_buf.value,
            alpha=self.cfg.alpha,
            beta=self.cfg.beta,
            gamma=self.cfg.gamma,
            delta=self.cfg.delta,
            epsilon=self.cfg.epsilon,
            tau=self.cfg.tau,
            focal_gamma=self.cfg.focal_gamma,
            focal_alpha=self.cfg.focal_alpha,
            label_smoothing=self.cfg.label_smoothing
        )
        loss_fn = self.loss_factory.build()
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(self.cfg.lr),
            loss=loss_fn,
            metrics=["accuracy"]
        )

    def _fit_one_cycle(self, cycle_idx: int) -> tf.keras.callbacks.History:
        print(f"\n=== Recycling Cycle {cycle_idx + 1}/{self.cfg.cycles} ===")
        es = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=self.cfg.patience, restore_best_weights=True
        )
        history = self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.cfg.epochs_per_cycle,
            batch_size=self.cfg.batch_size,
            validation_data=(self.x_val, self.y_val) if self.x_val is not None else None,
            callbacks=[es],
            shuffle=True,
            verbose=1,
        )
        return history

    def _update_prev_probs(self) -> None:
        # 전체 train에 대해 예측 → EMA update
        preds = self.model.predict(self.x_train, batch_size=self.cfg.batch_size, verbose=0)
        self.ema_buf.update(preds)

    def train(self) -> None:
        for c in range(self.cfg.cycles):
            hist = self._fit_one_cycle(c)
            self.history_per_cycle.append(hist.history)
            self._update_prev_probs()
            # prev_probs_var를 업데이트 했으므로 compile 없이 재사용 가능 (tf.Variable 참조)
            # 하지만 가중치(알파/베타 등)를 바꾸고 싶으면 재-compile 호출

    def evaluate(self) -> Tuple[float, float]:
        if self.x_val is None or self.y_val is None:
            raise ValueError("x_val / y_val must be provided to evaluate.")
        return self.model.evaluate(self.x_val, self.y_val, verbose=0)

    def plot_last_cycle(self) -> None:
        import matplotlib.pyplot as plt
        if not self.history_per_cycle:
            print("No training history to plot.")
            return
        hist = self.history_per_cycle[-1]
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(hist['loss'], label='train_loss')
        if 'val_loss' in hist:
            plt.plot(hist['val_loss'], label='val_loss')
        plt.legend();
        plt.title('Loss');
        plt.xlabel('epoch')
        plt.subplot(1, 2, 2)
        plt.plot(hist['accuracy'], label='train_acc')
        if 'val_accuracy' in hist:
            plt.plot(hist['val_accuracy'], label='val_acc')
        plt.legend();
        plt.title('Accuracy');
        plt.xlabel('epoch')
        plt.tight_layout();
        plt.show()


# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
@dataclass
class TrainConfig:
    alpha: float = 0.5
    beta: float = 0.2
    gamma: float = 0.1
    delta: float = 0.15
    epsilon: float = 0.05
    tau: float = 0.9
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    label_smoothing: float = 0.1
    ema_decay: float = 0.95
    cycles: int = 3
    epochs_per_cycle: int = 10
    batch_size: int = 64
    patience: int = 5
    lr: float = 1e-3
    csv_log_path: str = "logs/cifar100_loss.csv"


# 2) 데이터 로딩
loader = CIFAR100Loader(
    tar_path="cifar-100-python.tar.gz",
    label_mode="fine",
    normalize=True,
    one_hot=True,
    channels_last=True
)
(x_train, y_train), (x_test, y_test) = loader.load()

# 3) 모델 정의
model = models.Sequential([
    layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=x_train.shape[1:]),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.25),
    layers.Dense(y_train.shape[-1], activation='softmax')
])

# 4) 학습 구성 및 실행
cfg = TrainConfig()
trainer = RecyclingTrainer(
    model=model,
    x_train=x_train, y_train=y_train,
    x_val=x_test, y_val=y_test,
    cfg=cfg
)

trainer.train()
loss, acc = trainer.evaluate()
print(f"Final Val -> Loss: {loss:.4f}, Acc: {acc:.4f}")
trainer.plot_last_cycle()

# ---------------------------------------------------------------
# Demo: CIFAR-10
# ---------------------------------------------------------------
if __name__ == "__main__":
    from keras import datasets, utils, layers, models

    # 1) 데이터 로드
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0
    train_labels = utils.to_categorical(train_labels, 10)
    test_labels = utils.to_categorical(test_labels, 10)

    # 2) 간단한 CNN 모델
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=train_images.shape[1:]),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.25),
        layers.Dense(10, activation='softmax')
    ])

    # 3) Trainer 생성 & 학습
    cfg = TrainConfig(
        alpha=0.5, beta=0.2, gamma=0.1, delta=0.15, epsilon=0.05,
        tau=0.9, focal_gamma=2.0, focal_alpha=0.25,
        label_smoothing=0.1,
        ema_decay=0.95,
        cycles=3, epochs_per_cycle=10, batch_size=64,
        lr=1e-3, patience=5
    )

    trainer = RecyclingTrainer(
        model=model,
        x_train=train_images, y_train=train_labels,
        x_val=test_images, y_val=test_labels,
        cfg=cfg
    )

    trainer.train()
    loss, acc = trainer.evaluate()
    print(f"Final Val -> Loss: {loss:.4f}, Acc: {acc:.4f}")
    trainer.plot_last_cycle()