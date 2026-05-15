from __future__ import annotations

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.loggers import TensorBoardLogger
from spoofdet import config as config
# from .config import EARLY_STOPPING_LIMIT
# from .config import EPOCHS
# from .config import MODEL_NAME

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='checkpoints/',
    filename=config.MODEL_NAME + '_{epoch:02d}_{val_loss:.4f}',
    save_top_k=1,
    mode='min'
)

my_logger = CSVLogger(save_dir='logs/', name=str(config.MODEL_NAME) + '_run')
trainer = L.Trainer(accelerator='gpu',
                    devices=1, max_epochs=config.EPOCHS,
                    callbacks=[EarlyStopping(
                        monitor='val_loss',
                        patience=config.EARLY_STOPPING_LIMIT),
                        TQDMProgressBar(refresh_rate=20),
                        checkpoint_callback],
                    logger=[TensorBoardLogger('tb_logs', name=config.MODEL_NAME),
                            my_logger],
                    enable_progress_bar=True,
                    # precision="bf16",
                    # gradient_clip_val=1.0,
                    )
