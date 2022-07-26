import os

from tqdm import tqdm


if __name__ == "__main__":
    dataset_path = "pc_smsm"
    video_path = "smsm_otam"

    splits = ["train", "test", "val"]

    for split in splits:
        class_list = [c for c in os.listdir(video_path) if split in c]

        with open(os.path.join(dataset_path, "annotations", split + ".txt"), "w") as f:
            for label, c in enumerate(tqdm(class_list)):
                video_list = os.listdir(os.path.join(video_path, c))
                for video in video_list:
                    folder_name = video.split(".")[0]
                    num_frames = len(os.listdir(os.path.join(dataset_path, "frames", folder_name)))
                    f.write("frames/" + folder_name + " " + str(num_frames) + " " + str(label) + "\n")
                label += 1
