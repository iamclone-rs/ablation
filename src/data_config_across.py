ACROSS_DATASET_SEED = 42

# Sampled once with random.Random(42) from the existing 30 unseen classes
# already defined for each target dataset in src/data_config.py.
ACROSS_UNSEEN_CLASSES = {
    "tuberlin": [
        "windmill",
        "suitcase",
        "banana",
        "rollerblades",
        "fan",
        "canoe",
        "parachute",
        "streetlight",
        "bread",
        "ant",
        "tractor",
        "t-shirt",
        "lighter",
        "bus",
        "frying-pan",
        "brain",
        "snowboard",
        "horse",
        "space shuttle",
        "teacup",
        "pizza",
    ],
    "quickdraw": [
        "fire_hydrant",
        "mouse",
        "zebra",
        "megaphone",
        "cake",
        "fan",
        "windmill",
        "raccoon",
        "feather",
        "bread",
        "rhinoceros",
    ],
}
