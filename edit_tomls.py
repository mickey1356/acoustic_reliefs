import tomlkit
import os

# imgs = ["patchwork.jpeg", "floral_pattern.jpeg", "mountains5.jpeg", "waves_abstract2.png", "matterhorn2.jpg", "waves_abstract.png", "windows_bloom.jpg", "patterns.png", "stripes.png", "mondrian.png"]
bname = "configs/flowers.toml"

wts = [(1, 0), (1, 1/3), (1, 2/3), (1, 1), (2/3, 1/3), (2/3, 2/3), (2/3, 1), (1/3, 2/3), (1/3, 1), (0, 1)]


for i, (ac, ap) in enumerate(wts):
    with open(bname, "rb") as f:
        config = tomlkit.load(f)

    ac_wt = config["optimization"]["ac_wt"] * ac
    cl_wt = float(config["optimization"]["cl_wt"]) * ap
    vw_wt = [float(w) * ap for w in config["optimization"]["vw_wts"]]
    config["name"] = f"flowers_ac{ac_wt:.1f}_ap{ap:.1f}"
    config["optimization"]["ac_wt"] = ac_wt
    config["optimization"]["cl_wt"] = cl_wt
    config["optimization"]["vw_wts"] = vw_wt

    with open(f"configs/vary_weights_{i}.toml", "w") as f:
        tomlkit.dump(config, f)
    print(f'sbatch --ntasks=1 --cpus-per-task=32 --gpus=1 --mem-per-cpu=1G --time=8:0:0 --output={i}.txt --wrap="python acoustics_opt.py configs/vary_weights_{i}.toml"')

# for img in imgs:
#     ii = os.path.splitext(img)[0]
#     config["name"] = f"ptlight_0.9_{ii}"
#     config["optimization"]["tgt_fname"] = f"test-data/images/{img}"
#     with open(f"configs/{ii}_0.9.toml", "w") as f:
#         tomlkit.dump(config, f)
#     print(f'sbatch --ntasks=1 --cpus-per-task=32 --gpus=1 --mem-per-cpu=1G --time=32:0:0 --output={ii}.txt --wrap="python acoustics_opt.py configs/{ii}_0.9.toml"')
