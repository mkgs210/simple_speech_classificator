import os, time, math, csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchaudio.datasets import SPEECHCOMMANDS
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import RichProgressBar, EarlyStopping
from pytorch_lightning.profilers import SimpleProfiler
from pytorch_lightning.tuner import Tuner
import gc

# Collate-функция, которая дополняет аудиотензоры нулями до максимальной длины в батче
def custom_collate(batch):
    waveforms = [sample[0] for sample in batch]
    max_len = max(w.size(1) for w in waveforms)
    padded = []
    for sample in batch:
        waveform = sample[0]
        pad_len = max_len - waveform.size(1)
        if pad_len > 0:
            waveform = F.pad(waveform, (0, pad_len))
        padded.append((waveform,) + sample[1:])
    from torch.utils.data._utils.collate import default_collate
    return default_collate(padded)

# Датасет: оставляем только записи с метками "yes" и "no"
class YesNoCommands(SPEECHCOMMANDS):
    def __init__(self, root, subset):
        super().__init__(root, download=True)
        walker = self._walker  
        walker = [w for w in walker if os.path.exists(w)]
        self._walker = [w for w in walker if self._get_label(w).lower() in ["yes", "no"]]
        
    def _get_label(self, path):
        return os.path.basename(os.path.dirname(path))
    
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        waveform = sample[0]
        waveform = waveform.clone().detach().cpu().float().contiguous()
        return (waveform,) + sample[1:]

# DataModule с реальным разбиением: 80% для обучения, 20% для теста (и валидации)
class SpeechCommandsDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, split_ratio=0.8):
        super().__init__()
        self.data_dir = data_dir
        self.split_ratio = split_ratio
        self.batch_size = 128  # фиксированный размер батча

    def setup(self, stage=None):
        full_dataset = YesNoCommands(self.data_dir, subset="training")
        total = len(full_dataset)
        train_size = int(self.split_ratio * total)
        test_size = total - train_size
        self.train_set, self.test_set = random_split(full_dataset, [train_size, test_size])
        self.val_set = self.test_set
        print("Train set size:", len(self.train_set))
        print("Val set size:", len(self.val_set))
        print("Test set size:", len(self.test_set))
    
    def train_dataloader(self):
        return DataLoader(
            self.train_set, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=4,
            persistent_workers=True, 
            collate_fn=custom_collate
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, 
            batch_size=self.batch_size,
            num_workers=4,
            persistent_workers=True, 
            collate_fn=custom_collate
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, 
            batch_size=self.batch_size,
            num_workers=4,
            persistent_workers=True, 
            collate_fn=custom_collate
        )

# Импортируем наш слой LogMelFilterBanks
from melbanks_orig import LogMelFilterBanks

# Простая CNN-модель
class SimpleCNN(pl.LightningModule):
    def __init__(self, n_mels=80, conv_groups=1, lr=1e-3, batch_size=64):
        super().__init__()
        self.save_hyperparameters()
        self.feature_extractor = LogMelFilterBanks(n_mels=n_mels)
        self.conv_groups = conv_groups
        self.conv1 = nn.Conv2d(conv_groups, 16 * conv_groups, kernel_size=3, padding=1, groups=conv_groups)
        self.conv2 = nn.Conv2d(16 * conv_groups, 32 * conv_groups, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2)
        self.fc    = nn.Linear(32 * conv_groups, 2)
        self.lr = lr

    def forward(self, x):
        x = self.feature_extractor(x)         # (batch, n_mels, time)
        x = x.unsqueeze(1).repeat(1, self.conv_groups, 1, 1)  # (batch, conv_groups, n_mels, time)
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.mean(x, dim=[2,3])
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        waveform, sr, label, *_ = batch
        logits = self(waveform)
        target = torch.tensor([1 if l.lower() == "yes" else 0 for l in label], device=self.device)
        loss = F.cross_entropy(logits, target)
        bs = getattr(self.hparams, "batch_size", 64)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=bs)
        return loss

    def validation_step(self, batch, batch_idx):
        waveform, sr, label, *_ = batch
        logits = self(waveform)
        target = torch.tensor([1 if l.lower() == "yes" else 0 for l in label], device=self.device)
        loss = F.cross_entropy(logits, target)
        acc = (torch.argmax(logits, dim=1) == target).float().mean()
        bs = getattr(self.hparams, "batch_size", 64)
        self.log("val_loss", loss, prog_bar=True, batch_size=bs)
        self.log("val_acc", acc, prog_bar=True, batch_size=bs)

    def test_step(self, batch, batch_idx):
        waveform, sr, label, *_ = batch
        logits = self(waveform)
        target = torch.tensor([1 if l.lower() == "yes" else 0 for l in label], device=self.device)
        acc = (torch.argmax(logits, dim=1) == target).float().mean()
        bs = getattr(self.hparams, "batch_size", 64)
        self.log("test_acc", acc, batch_size=bs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def on_train_epoch_start(self):
        self.epoch_start = time.time()
        #print(f"=== Начало тренировочной эпохи {self.current_epoch} ===")

    def on_train_epoch_end(self):
        epoch_time = time.time() - self.epoch_start
        #print(f"=== Тренировочная эпоха {self.current_epoch} завершена за {epoch_time:.2f} секунд ===")
        self.log("epoch_time", epoch_time)

    def on_validation_epoch_end(self):
        #print(f"=== Валидационная эпоха {self.current_epoch} завершена ===")
        pass

    def on_test_epoch_end(self):
        #print("=== Тестирование завершено ===")
        pass

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def compute_flops(self, input_size=(1, 16000)):
        from ptflops import get_model_complexity_info
        macs, params = get_model_complexity_info(self, input_size, as_strings=True, print_per_layer_stat=False)
        return macs, params

if __name__ == "__main__":
    os.makedirs("./SpeechCommandsData", exist_ok=True)
    data_dir = "./SpeechCommandsData"
    
    group_values = [2, 4, 8, 16]
    n_mels_values = [20, 40, 80]
    results = {}
    
    for groups in group_values:
        for n_mels in n_mels_values:
            exp_name = f"groups_{groups}_mels_{n_mels}"
            logger = TensorBoardLogger("tb_logs", name=exp_name)
            
            dm = SpeechCommandsDataModule(data_dir)
            dm.setup()
            model = SimpleCNN(n_mels=n_mels, conv_groups=groups)
            
            callbacks = [
                RichProgressBar(),
                EarlyStopping(monitor="val_loss", patience=10, min_delta=0.01),
            ]
            
            trainer = pl.Trainer(
                max_epochs=300,
                accelerator="auto",
                logger=logger,
                callbacks=callbacks,
                #profiler="simple"#SimpleProfiler()
            )
            
            # Замеряем время
            start_time = time.time()
            trainer.fit(model, dm)
            epoch_time = (time.time() - start_time) / trainer.current_epoch
            
            # Тестирование
            test_results = trainer.test(model, dm)
            test_acc = test_results[0]['test_acc']
            
            # Подсчет параметров и FLOPS
            num_params = model.count_parameters()
            flops = model.compute_flops()
            
            results[(groups, n_mels)] = {
                'test_acc': test_acc,
                'num_params': num_params,
                'flops': flops,
                'epoch_time': epoch_time
            }
            
            # Очищаем память GPU после каждого эксперимента
            del model
            del trainer
            torch.cuda.empty_cache()
            gc.collect()

    with open("results.csv", "w", newline="") as csvfile:
        fieldnames = ["conv_groups", "n_mels", "test_acc", "num_params", "flops", "epoch_time"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for (g, nm), res in results.items():
            writer.writerow({
                "conv_groups": g,
                "n_mels": nm,
                "test_acc": res["test_acc"],
                "num_params": res["num_params"],
                "flops": res["flops"],
                "epoch_time": res["epoch_time"]
            })
