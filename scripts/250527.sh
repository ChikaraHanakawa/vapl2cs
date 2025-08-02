#!/bin/sh
set -euxo pipefail

sliding_csv_train="data/splits_multi/sliding_window_dset/train.csv"
sliding_csv_val="data/splits_multi/sliding_window_dset/val.csv"

python vap/data/datamodule.py --csv "$sliding_csv_train" --batch_size 4 --num_workers 8 --prefetch_factor 2
python vap/data/datamodule.py --csv "$sliding_csv_val" --batch_size 4 --num_workers 8 --prefetch_factor 2

python vap/main.py datamodule.train_path=$sliding_csv_train datamodule.val_path=$sliding_csv_val datamodule.batch_size=4 datamodule.num_workers=4 trainer.max_epochs=100
