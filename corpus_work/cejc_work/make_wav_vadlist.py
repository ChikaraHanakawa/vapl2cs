import os
import pandas as pd
import glob
import soundfile as sf
import numpy as np
import json

class InternalError(Exception):
    pass


def make_wav(kaiwa_id, cejc_wav_top_dir, vap_data_top_dir):
    # 会話IDからwavファイルのディレクトリを取得
    kaiwa_id_parts = kaiwa_id.split("_")
    kaiwa_dir = kaiwa_id_parts[0]
    kaiwa_id_dir = kaiwa_id[:8]
    wav_dir = os.path.join(cejc_wav_top_dir, "data", kaiwa_dir, kaiwa_id_dir)

    # wavファイルのリストを取得
    wav_path_list = sorted(glob.glob(os.path.join(wav_dir, f"{kaiwa_id}_IC[0-9][0-9].wav")))

    # assert len(wav_path_list) == 2, f"wavファイルが2つでない: {wav_dir}"
    if len(wav_path_list) != 2:
        print(f"wavファイルが2つでない: {wav_dir}")
        raise InternalError("wavファイルが2つでない")

    wav_data_list = []
    speaker_id_list = []
    for wav_path in wav_path_list:
        wav_data, sr = sf.read(wav_path)
        wav_data_list.append(wav_data)
        speaker_id = os.path.splitext(wav_path)[0][-4:]
        speaker_id_list.append(speaker_id)
    
    # wavデータをチャネル方向に結合
    wav_data = np.stack(wav_data_list).T

    # wavファイルの書き出し
    output_wav_data_dir = os.path.join(vap_data_top_dir, "wav")
    os.makedirs(output_wav_data_dir, exist_ok=True)

    output_wav_path = os.path.join(output_wav_data_dir, f"{kaiwa_id}.wav")
    sf.write(output_wav_path, wav_data, sr)

    return speaker_id_list


def make_vad(kaiwa_id, speaker_ids, cejc2304_top_dir, vap_data_top_dir):
    # 会話IDからtransUnit.csvのパスを取得
    kaiwa_id_parts = kaiwa_id.split("_")
    kaiwa_dir = kaiwa_id_parts[0]
    kaiwa_id_dir = kaiwa_id[:8]
    trans_unit_csv_path = os.path.join(cejc2304_top_dir, "data", kaiwa_dir, kaiwa_id_dir, kaiwa_id +"-transUnit.csv")

    df = pd.read_csv(trans_unit_csv_path, encoding="shift-jis")

    # 話者IDをチャネルIDに変換する辞書
    sid2channel = {speaker_id: i for i, speaker_id in enumerate(speaker_ids)}

    # import ipdb; ipdb.set_trace()

    vad_data = [[], []]
    reported_speaker_ids = []
    for i, row in df.iterrows():
        speakerID = row["speakerID"][:4] # 最初の4文字だけ取得
        if speakerID not in sid2channel:
            if speakerID not in reported_speaker_ids:
                print("speakerID not in sid2channel:", speakerID)
                reported_speaker_ids.append(speakerID)
            continue
        channel = sid2channel[speakerID]
        vad_data[channel].append([float(row["startTime"]), float(row["endTime"])])
    
    # vadデータで，前の区間の終了時間と次の区間の開始時間が同じ場合は結合する
    for i in range(2):
        new_vad_data = []
        for j in range(len(vad_data[i])):
            if j == 0:
                new_vad_data.append(vad_data[i][j])
            else:
                if vad_data[i][j-1][1] == vad_data[i][j][0]:
                    new_vad_data[-1][1] = vad_data[i][j][1]
                else:
                    new_vad_data.append(vad_data[i][j])
        vad_data[i] = new_vad_data



    # vadデータの書き出し
    output_vad_data_dir = os.path.join(vap_data_top_dir, "vad_list")
    os.makedirs(output_vad_data_dir, exist_ok=True)
    output_vad_path = os.path.join(output_vad_data_dir, f"{kaiwa_id}.json")
    with open(output_vad_path, "w") as f:
        json.dump(vad_data, f)


def run(cejc2304_top_dir, cejc_wav_top_dir, vap_data_top_dir):
    # 会話.csv の読み込み
    kaiwa_csv_path = os.path.join(cejc2304_top_dir, 'metaInfo', '個別ファイル', '会話.csv')
    kaiwa_df = pd.read_csv(kaiwa_csv_path, encoding='shift-jis')

    # 話者数2の行だけを抽出
    kaiwa_df = kaiwa_df[kaiwa_df['話者数'] == 2]

    # 会話IDのリストを作成
    kaiwa_id_list = kaiwa_df['会話ID'].tolist()

    # 会話IDごとに処理
    for kaiwa_id in kaiwa_id_list:
        print(kaiwa_id)

        try:
            # wavの書き出しと話者IDの取得
            speaker_ids = make_wav(kaiwa_id, cejc_wav_top_dir, vap_data_top_dir)

            # vad情報を書き出し
            make_vad(kaiwa_id, speaker_ids,cejc2304_top_dir, vap_data_top_dir)
        except InternalError as e:
            print(e)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("cejc2304_top_dir", type=str)
    parser.add_argument("cejc_wav_top_dir", type=str)
    parser.add_argument("vap_data_top_dir", type=str)

    args = parser.parse_args()

    run(args.cejc2304_top_dir, args.cejc_wav_top_dir, args.vap_data_top_dir)