import os
import glob
import json

def make_vad(data_file_path):
    vads = [[], []]
    with open(data_file_path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            # 8行分データを読み込む
            data = [line.rstrip()]
            for i in range(7):
                data.append(f.readline().rstrip())
            # assert data[7] == "#", f"Data format corrupted: {data[7]}"
            if data[7] != "#":
                print(f"WARNING: Data format corrupted: {data[7]}")
            
            channel = 0 if data[2] == "B" else 1
            try:
                ts = int(data[3])/1000
            except ValueError:
                print(f"WARNING: {data[3]}")
                ts = int(data[3][:-1])/1000
            try:
                te = int(data[4])/1000
            except ValueError:
                print(f"WARNING: {data[4]}")
                te = int(data[4][:-1])/1000

            vad = [ts, te]
            vads[channel].append(vad)
    return vads

def run(input_dir, output_dir):
    data_files = sorted(glob.glob(os.path.join(input_dir, "**/*.dat"), recursive=True))
    for data_file_path in data_files:
        output_json_path = os.path.join(output_dir, os.path.basename(data_file_path).replace(".dat", ".json"))
        print(f"{data_file_path} -> {output_json_path}")

        vads = make_vad(data_file_path)

        if not os.path.exists(os.path.dirname(output_json_path)):
            os.makedirs(os.path.dirname(output_json_path))

        with open(output_json_path, "w") as f:
            json.dump(vads, f)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", "-i", type=str, action="append")
    parser.add_argument("output_dir", type=str)

    args = parser.parse_args()

    for input_dir in args.input_dir:
        run(input_dir, args.output_dir)