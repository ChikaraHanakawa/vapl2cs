#!/bin/sh
set -euxo pipefail

corpus=(cejc waseda_soma csj rwcp pasd uudb tabidachi)

for c in ${corpus[@]}; do
    python scripts/fixed_ratio_selector.py data/splits_${c}/audio_vad.csv data/splits_${c}/0430_exp/audio_vad/ --train 0.8 --dev 0.05 --test 0.15
done

for c in ${corpus[@]}; do
    audio_vad_csv_train="data/splits_${c}/0430_exp/audio_vad/train.csv"
    audio_vad_csv_val="data/splits_${c}/0430_exp/audio_vad/val.csv"
    audio_vad_csv_test="data/splits_${c}/0430_exp/audio_vad/test.csv"
    sliding_csv_train="data/splits_${c}/0430_exp/sliding_window_dset/train.csv"
    sliding_csv_val="data/splits_${c}/0430_exp/sliding_window_dset/val.csv"
    sliding_csv_test="data/splits_${c}/0430_exp/sliding_window_dset/test.csv"
    audio_vad_hs_train="data/splits_${c}/0430_exp/audio_vad_hs/train.csv"
    audio_vad_hs_val="data/splits_${c}/0430_exp/audio_vad_hs/val.csv"
    audio_vad_hs_test="data/splits_${c}/0430_exp/audio_vad_hs/test.csv"
    python vap/data/create_sliding_window_dset.py --audio_vad_csv "$audio_vad_csv_train" --output "$sliding_csv_train" --duration 20 --overlap 5 --horizon 2
    python vap/data/create_sliding_window_dset.py --audio_vad_csv "$audio_vad_csv_val" --output "$sliding_csv_val" --duration 20 --overlap 5 --horizon 2
    python vap/data/create_sliding_window_dset.py --audio_vad_csv "$audio_vad_csv_test" --output "$sliding_csv_test" --duration 20 --overlap 5 --horizon 2
    python vap/data/datamodule.py --csv "$sliding_csv_train" --batch_size 4 --num_workers 8 --prefetch_factor 2
    python vap/data/datamodule.py --csv "$sliding_csv_val" --batch_size 4 --num_workers 8 --prefetch_factor 2
    python vap/data/datamodule.py --csv "$sliding_csv_test" --batch_size 4 --num_workers 8 --prefetch_factor 2
    python vap/data/dset_event.py --audio_vad_csv "$audio_vad_csv_train" --output "$audio_vad_hs_train" --pre_cond_time 1 --post_cond_time 2 --min_silence_time 0.1
    python vap/data/dset_event.py --audio_vad_csv "$audio_vad_csv_val" --output "$audio_vad_hs_val" --pre_cond_time 1 --post_cond_time 2 --min_silence_time 0.1
    python vap/data/dset_event.py --audio_vad_csv "$audio_vad_csv_test" --output "$audio_vad_hs_test" --pre_cond_time 1 --post_cond_time 2 --min_silence_time 0.1
done

large=(tabidachi cejc waseda_soma)
small=(csj rwcp pasd uudb)
log_file=logs/train.log

for l in ${large[@]}; do
    train_path="data/splits_${l}/0430_exp/sliding_window_dset/train.csv"
    val_path="data/splits_${l}/0430_exp/sliding_window_dset/val.csv"
    echo "`date '+%Y-%m-%d %H:%M:%S'` - Corpus: $l" >> $log_file
    python vap/main.py datamodule.train_path=$train_path datamodule.val_path=$val_path datamodule.batch_size=4 datamodule.num_workers=4 trainer.max_epochs=100
    echo "Finish: $l"
    for s in ${small[@]}; do
        train_path="data/splits_${l}/0430_exp/sliding_window_dset/${s}_add/train.csv"
        val_path="data/splits_${l}/0430_exp/sliding_window_dset/${s}_add/val.csv"
        echo "`date '+%Y-%m-%d %H:%M:%S'` - Corpus: $l&$s" >> $log_file
        python vap/main.py datamodule.train_path=$train_path datamodule.val_path=$val_path datamodule.batch_size=4 datamodule.num_workers=4 trainer.max_epochs=100
        echo "Finish: $l&$s"
    done
done

base=(tabidachi cejc waseda_soma)
model=(csj rwcp pasd uudb)
test=(tabidachi cejc waseda_soma csj rwcp pasd uudb)

for b in ${base[@]}; do
    for t in ${test[@]}; do
        python vap/eval_events.py --checkpoint ckpt/250501_ckpt/${b}/${b}.ckpt --csv data/splits_"$t"/0430_exp/audio_vad_hs/test.csv --output results/250501_results/$b/$b/$t/hs/ --plot
        python vap/eval_events_predhs.py --checkpoint ckpt/250501_ckpt/${b}/${b}.ckpt --csv data/splits_"$t"/0430_exp/audio_vad_predhs/test.csv --output results/250501_results/$b/$b/$t/predhs/ --plot
        python vap/eval_events_bc.py --checkpoint ckpt/250501_ckpt/${b}/${b}.ckpt --csv data/splits_"$t"/0430_exp/audio_vad_bc/test.csv --output results/250501_results/$b/$b/$t/bc/ --plot
        for m in ${model[@]}; do
            python vap/eval_events.py --checkpoint ckpt/250501_ckpt/${b}/${m}.ckpt --csv data/splits_"$t"/0430_exp/audio_vad_hs/test.csv --output results/250501_results/$b/$m/$t/hs/ --plot
            python vap/eval_events_predhs.py --checkpoint ckpt/250501_ckpt/${b}/${m}.ckpt --csv data/splits_"$t"/0430_exp/audio_vad_predhs/test.csv --output results/250501_results/$b/$m/$t/predhs/ --plot
            python vap/eval_events_bc.py --checkpoint ckpt/250501_ckpt/${b}/${m}.ckpt --csv data/splits_"$t"/0430_exp/audio_vad_bc/test.csv --output results/250501_results/$b/$m/$t/bc/ --plot
        done
    done
done

python scripts/combine_csv.py

for m in ${model[@]}; do
    python vap/data/dset_event.py --audio_vad_csv data/splits_${m}/audio_vad.csv --output data/splits_${m}/0430_exp/audio_vad_hs/test_all.csv --pre_cond_time 1 --post_cond_time 2 --min_silence_time 0.1
    python vap/data/dset_event_predhs.py --audio_vad_csv data/splits_${m}/audio_vad.csv --output data/splits_${m}/0430_exp/audio_vad_predhs/test_all.csv --pre_cond_time 1 --post_cond_time 2 --min_silence_time 0.1
    python vap/data/dset_event_bc.py --audio_vad_csv data/splits_${m}/audio_vad.csv --output data/splits_${m}/0430_exp/audio_vad_bc/test_all.csv --pre_cond_time 1 --post_cond_time 2 --min_silence_time 0.1
done

