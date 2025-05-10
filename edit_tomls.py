import tomlkit
import os

imgs = ["patchwork.jpeg", "floral_pattern.jpeg", "mountains5.jpeg", "waves_abstract2.png", "matterhorn2.jpg", "waves_abstract.png", "windows_bloom.jpg", "patterns.png", "stripes.png", "mondrian.png"]
bname = "configs/ptlight_0.9.toml"

with open(bname, "rb") as f:
    config = tomlkit.load(f)

for img in imgs:
    ii = os.path.splitext(img)[0]
    config["name"] = f"ptlight_0.9_{ii}"
    config["optimization"]["tgt_fname"] = f"test-data/images/{img}"
    with open(f"configs/{ii}_0.9.toml", "w") as f:
        tomlkit.dump(config, f)