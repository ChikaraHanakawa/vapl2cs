#!/bin/sh
set -euxo pipefail

hour=(10h 20h 30h 40h 50h 60h 70h 80h 90h 100h)
corpus=(cejc waseda_soma)

for c in ${corpus[@]}; do
    audio_vad_csv_val="data/splits_${c}/duration_splits/audio_vad/dev_3h.csv"
    audio_vad_csv_test="data/splits_${c}/duration_splits/audio_vad/test_10h.csv"
    sliding_csv_val="data/splits_${c}/duration_splits/sliding_window_dset/dev_3h.csv"
    sliding_csv_test="data/splits_${c}/duration_splits/sliding_window_dset/test_10h.csv"
    audio_vad_hs_val="data/splits_${c}/duration_splits/audio_vad_hs/dev_3h.csv"
    audio_vad_hs_test="data/splits_${c}/duration_splits/audio_vad_hs/test_10h.csv"
    python vap/data/create_sliding_window_dset.py --audio_vad_csv "$audio_vad_csv_val" --output "$sliding_csv_val" --duration 20 --overlap 5 --horizon 2
    python vap/data/create_sliding_window_dset.py --audio_vad_csv "$audio_vad_csv_test" --output "$sliding_csv_test" --duration 20 --overlap 5 --horizon 2
    python vap/data/datamodule.py --csv "$sliding_csv_val" --batch_size 4 --num_workers 8 --prefetch_factor 2
    python vap/data/datamodule.py --csv "$sliding_csv_test" --batch_size 4 --num_workers 8 --prefetch_factor 2
    python vap/data/dset_event.py --audio_vad_csv "$audio_vad_csv_val" --output "$audio_vad_hs_val" --pre_cond_time 1 --post_cond_time 2 --min_silence_time 0.1
    python vap/data/dset_event.py --audio_vad_csv "$audio_vad_csv_test" --output "$audio_vad_hs_test" --pre_cond_time 1 --post_cond_time 2 --min_silence_time 0.1
done

for c in ${corpus[@]}; do
    if [ "$c" = "cejc" ]; then
        for h in ${hour[@]:0:7}; do
            audio_vad_csv_train="data/splits_${c}/duration_splits/audio_vad/train_${h}.csv"
            sliding_csv_train="data/splits_${c}/duration_splits/sliding_window_dset/train_${h}.csv"
            audio_vad_hs_train="data/splits_${c}/duration_splits/audio_vad_hs/train_${h}.csv"
            python vap/data/create_sliding_window_dset.py --audio_vad_csv "$audio_vad_csv_train" --output "$sliding_csv_train" --duration 20 --overlap 5 --horizon 2
            python vap/data/datamodule.py --csv "$sliding_csv_train" --batch_size 4 --num_workers 8 --prefetch_factor 2
            python vap/data/dset_event.py --audio_vad_csv "$audio_vad_csv_train" --output "$audio_vad_hs_train" --pre_cond_time 1 --post_cond_time 2 --min_silence_time 0.1
        done
    else
        for h in ${hour[@]:0:3}; do
            audio_vad_csv_train="data/splits_${c}/duration_splits/audio_vad/train_${h}.csv"
            sliding_csv_train="data/splits_${c}/duration_splits/sliding_window_dset/train_${h}.csv"
            audio_vad_hs_train="data/splits_${c}/duration_splits/audio_vad_hs/train_${h}.csv"
            python vap/data/create_sliding_window_dset.py --audio_vad_csv "$audio_vad_csv_train" --output "$sliding_csv_train" --duration 20 --overlap 5 --horizon 2
            python vap/data/datamodule.py --csv "$sliding_csv_train" --batch_size 4 --num_workers 8 --prefetch_factor 2
            python vap/data/dset_event.py --audio_vad_csv "$audio_vad_csv_train" --output "$audio_vad_hs_train" --pre_cond_time 1 --post_cond_time 2 --min_silence_time 0.1
        done
    fi
done

log_file=logs/train.log

for c in ${corpus[@]}; do
    if [ "$c" = "cejc" ]; then
        for h in ${hour[@]:0:7}; do
            train_path=data/splits_${c}/duration_splits/sliding_window_dset/train_${h}.csv
            val_path=data/splits_${c}/duration_splits/sliding_window_dset/dev_3h.csv

            python vap/main.py datamodule.train_path=$train_path datamodule.val_path=$val_path datamodule.batch_size=4 datamodule.num_workers=4 trainer.max_epochs=100
            echo "`date '+%Y-%m-%d %H:%M:%S'` - TrainData-time: $h" >> $log_file
            echo "Finish: $c-$h"
        done
    else
        for h in ${hour[@]:0:3}; do
            train_path=data/splits_${c}/duration_splits/sliding_window_dset/train_${h}.csv
            val_path=data/splits_${c}/duration_splits/sliding_window_dset/dev_3h.csv

            python vap/main.py datamodule.train_path=$train_path datamodule.val_path=$val_path datamodule.batch_size=4 datamodule.num_workers=4 trainer.max_epochs=100
            echo "`date '+%Y-%m-%d %H:%M:%S'` - TrainData-time: $h" >> $log_file
            echo "Finish: $c-$h"
        done
    fi
done
