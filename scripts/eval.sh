#!/bin/sh
set -euxo pipefail

# hour=("10h" "20h" "30h" "40h" "50h" "60h" "70h" "80h" "90h" "100h")
# corpus=(cejc waseda_soma)

# for c in "${corpus[@]}"; do
#     if [ "$c" = "cejc" ]; then
#         for h in "${hour[@]:0:7}"; do
#             echo "Evaluating $c with $h"
#             python vap/eval_events.py --checkpoint ckpt/250429_ckpt/${c}_${h}.ckpt --csv data/splits_"$c"/time_separate_SlidingWindowDset/audio_vad_hs/test.csv --output results/250429_results/cejc/"$h"/ --plot
#         done
#     elif [ "$c" = "waseda_soma" ]; then
#         for h in "${hour[@]:0:4}"; do
#             echo "Evaluating $c with $h"
#             python vap/eval_events.py --checkpoint ckpt/250429_ckpt/${c}_${h}.ckpt --csv data/splits_"$c"/time_separate_SlidingWindowDset/audio_vad_hs/test.csv --output results/250429_results/waseda_soma/"$h"/ --plot
#         done
#     fi
# done

without=(tabidachi cejc waseda_soma)
test=(tabidachi cejc waseda_soma csj rwcp pasd uudb)
for w in "${without[@]}"; do
    for t in "${test[@]}"; do
        echo "Evaluating without $w : Test $t"
        if [ "$t" = "tabidachi" ] || [ "$t" = "cejc" ] || [ "$t" = "waseda_soma" ]; then
            python vap/eval_events.py --checkpoint ckpt/250501_ckpt/${w}_without_multi.ckpt --csv data/splits_${t}/0430_exp/audio_vad_hs/test.csv --output results/250503_results/without_$w/$t/hs/ --plot
            python vap/eval_events_predhs.py --checkpoint ckpt/250501_ckpt/${w}_without_multi.ckpt --csv data/splits_${t}/0430_exp/audio_vad_predhs/test.csv --output results/250503_results/without_$w/$t/predhs/ --plot
            python vap/eval_events_bc.py --checkpoint ckpt/250501_ckpt/${w}_without_multi.ckpt --csv data/splits_${t}/0430_exp/audio_vad_bc/test.csv --output results/250503_results/without_$w/$t/bc/ --plot
            continue
        fi
        python vap/eval_events.py --checkpoint ckpt/250501_ckpt/${w}_without_multi.ckpt --csv data/splits_${t}/0430_exp/audio_vad_hs/test_all.csv --output results/250503_results//without_$w/$t/hs/ --plot
        python vap/eval_events_predhs.py --checkpoint ckpt/250501_ckpt/${w}_without_multi.ckpt --csv data/splits_${t}/0430_exp/audio_vad_predhs/test_all.csv --output results/250503_results/without_$w/$t/predhs/ --plot
        python vap/eval_events_bc.py --checkpoint ckpt/250501_ckpt/${w}_without_multi.ckpt --csv data/splits_${t}/0430_exp/audio_vad_bc/test_all.csv --output results/250503_results//without_$w/$t/bc/ --plot
    done
done

for t in "${test[@]}"; do
    echo "Evaluating Multi : Test $t"
    python vap/eval_events.py --checkpoint ckpt/250501_ckpt/multi.ckpt --csv data/splits_${t}/0430_exp/audio_vad_hs/test.csv --output results/250503_results/multi/$t/hs/ --plot
    python vap/eval_events_predhs.py --checkpoint ckpt/250501_ckpt/multi.ckpt --csv data/splits_${t}/0430_exp/audio_vad_predhs/test.csv --output results/250503_results/multi/$t/predhs/ --plot
    python vap/eval_events_bc.py --checkpoint ckpt/250501_ckpt/multi.ckpt --csv data/splits_${t}/0430_exp/audio_vad_bc/test.csv --output results/250503_results/multi/$t/bc/ --plot
done