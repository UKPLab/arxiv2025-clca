import json


def filter_dialog(input_file, output_file):
    good_dialog = []
    with open(input_file, 'r') as f:
        for line in f.readlines():
            data = json.loads(line.strip())
            # filter out the bad quality ones
            if data["meta_quality"] == "bad" and data["meta_quality_confidence"] == "very confident":
                continue
            if data["quality"] == "bad" and data["quality_confidence"] == "very confident":
                continue
            good_dialog.append(data)

    print("good_dialog", len(good_dialog))
    with open(output_file, 'w') as f:
        for d in good_dialog:
            f.write(json.dumps(d) + "\n")

