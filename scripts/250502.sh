#!/bin/sh
set -euxo pipefail

model=(tabidachi cejc waseda_soma)
log_file=./logs/train.log

train_path=./data/splits_multi/sliding_window_dset/train.csv
val_path=./data/splits_multi/sliding_window_dset/val.csv
echo "`date '+%Y-%m-%d %H:%M:%S'` - Model: Multi" >> $log_file
python vap/main.py datamodule.train_path=$train_path datamodule.val_path=$val_path datamodule.batch_size=4 datamodule.num_workers=4 trainer.max_epochs=100
for m in ${model[@]}; do
    train_path=./data/without_multi/${m}_without_train.csv
    val_path=./data/without_multi/${m}_without_val.csv
    echo "`date '+%Y-%m-%d %H:%M:%S'` - Without: $m" >> $log_file
    python vap/main.py datamodule.train_path=$train_path datamodule.val_path=$val_path datamodule.batch_size=4 datamodule.num_workers=4 trainer.max_epochs=100
done