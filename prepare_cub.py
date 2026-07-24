from pathlib import Path
import shutil


def read_image_paths(dataset_root):
    image_paths = {}
    with open(dataset_root / "images.txt", "r", encoding="utf-8") as file:
        for line in file:
            image_id, relative_path = line.strip().split(maxsplit=1)
            image_paths[image_id] = relative_path
    return image_paths


def read_split_flags(dataset_root):
    split_flags = {}
    with open(dataset_root / "train_test_split.txt", "r", encoding="utf-8") as file:
        for line in file:
            image_id, is_train = line.strip().split()
            split_flags[image_id] = is_train == "1"
    return split_flags


def prepare_cub():
    dataset_root = Path("data/datasets/CUB_200_2011")
    output_root = Path("data/datasets/cub")

    if not dataset_root.exists():
        raise FileNotFoundError(
            "Missing data/datasets/CUB_200_2011. "
            "Extract CUB_200_2011.tgz into data/datasets first."
        )

    image_paths = read_image_paths(dataset_root)
    split_flags = read_split_flags(dataset_root)

    train_count = 0
    test_count = 0

    for image_id, relative_path in image_paths.items():
        split_name = "train" if split_flags[image_id] else "test"
        relative_path = Path(relative_path)
        class_name = relative_path.parent.name

        source_file = dataset_root / "images" / relative_path
        target_file = output_root / split_name / class_name / relative_path.name

        target_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target_file)

        if split_name == "train":
            train_count += 1
        else:
            test_count += 1

    print("CUB200 prepared.")
    print(f"Train images: {train_count}")
    print(f"Test images: {test_count}")
    print(f"Output: {output_root}")


if __name__ == "__main__":
    prepare_cub()
