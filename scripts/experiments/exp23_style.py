"""
One look for every experiment 23 / learnlet figure: hue = where a dictionary's filters come from (the data it was
trained on), line style + marker = the variant (per-scale or shared filters, extrapolated, learnt thresholds).
Categorical hues of the dataviz reference palette; checked per panel (OKLab, Machado 2009 CVD): the ImageNet panels
(aqua, blue, orange) clear every gate, the FIRE panels (aqua, blue, red, green, yellow) clear the normal-vision floor
(>= 15.6) with CVD separations 6.9-7.6, legal only with the secondary encoding here (markers, line styles) plus
direct labels -- keep both. Neutral ink for references (the pipeline, the S = 0 ablation).
"""
HUE = dict(starlet="#1baf7a", imagenet="#2a78d6", network="#eb6834", fire400="#e34948", fire600="#008300",
           fireaug="#eda100", pipeline="#52514e", ablation="#9a9893")
INK, MUTED = "#0b0b0b", "#52514e"

# (dictionary, threshold rule) -> label, colour, line style, marker
STYLE = {
    ("starlet", "pdtau"): ("pipeline: starlet, soft + reweighting", HUE["pipeline"], "--", "x"),
    ("starlet", "noise"): ("starlet", HUE["starlet"], "-", "o"),
    ("starlet", "learnt"): ("starlet at the network's thresholds (S = 0)", HUE["ablation"], ":", "v"),
    ("learnlet", "noise"): ("ImageNet learnlet filters", HUE["imagenet"], "-", "o"),
    ("learnlet_ext", "noise"): ("ImageNet filters, scale 3 extrapolated to 4-6", HUE["imagenet"], "--", "D"),
    ("learnlet", "learnt"): ("ImageNet learnlet network (learnt thresholds)", HUE["network"], "-", "P"),
    ("learnlet_fire", "noise"): ("FIRE 256 px, per scale (4 scales)", HUE["fire400"], "-", "o"),
    ("learnlet_fire_shared4", "noise"): ("FIRE 256 px, shared (4 trained)", HUE["fire400"], "--", "s"),
    ("learnlet_fire_shared6", "noise"): ("FIRE 256 px, shared (6 trained)", HUE["fire400"], ":", "^"),
    ("learnlet_fire_n600", "noise"): ("FIRE 600 px, per scale (7 scales)", HUE["fire600"], "-", "o"),
    ("learnlet_fire_n600_shared", "noise"): ("FIRE 600 px, shared (7 trained)", HUE["fire600"], "--", "s"),
    ("learnlet_fire_aug", "noise"): ("FIRE scale-augmented, per scale", HUE["fireaug"], "-", "o"),
    ("learnlet_fire_aug_shared", "noise"): ("FIRE scale-augmented, shared", HUE["fireaug"], "--", "s"),
}
for d in ("learnlet_fire", "learnlet_fire_shared6", "learnlet_fire_n600", "learnlet_fire_n600_shared",
          "learnlet_fire_aug", "learnlet_fire_aug_shared"):           # the FIRE models with their learnt thresholds
    lab, c, ls, m = STYLE[(d, "noise")]
    STYLE[(d, "learnt")] = (lab + ", learnt thresholds", c, ls, "P")

# the pipeline's own method (pdtau: primal-dual, soft thresholding + reweighting) with the dictionary swapped
STYLE.update({("learnlet", "pdtau"): ("pipeline with the ImageNet learnlet", HUE["imagenet"], "-", "X"),
              ("learnlet_ext32", "pdtau"): ("pipeline with ImageNet extrapolated (top 32)", HUE["imagenet"], "--", "X"),
              ("learnlet_fire_n600", "pdtau"): ("pipeline with FIRE 600 px, per scale", HUE["fire600"], "-", "X"),
              ("learnlet_fire_shared4", "pdtau"): ("pipeline with FIRE 256 px, shared", HUE["fire400"], "--", "X")})
# short names for image-panel titles
SHORT = {("starlet", "pdtau"): "Pipeline", ("starlet", "noise"): "Starlet", ("starlet", "learnt"): "Starlet, S = 0",
         ("learnlet", "noise"): "ImageNet filters", ("learnlet_ext", "noise"): "ImageNet, extrapolated",
         ("learnlet", "learnt"): "ImageNet network", ("learnlet_fire", "noise"): "FIRE 256, per scale",
         ("learnlet_fire_shared4", "noise"): "FIRE 256, shared 4", ("learnlet_fire_shared6", "noise"): "FIRE 256, shared 6",
         ("learnlet_fire_n600", "noise"): "FIRE 600, per scale", ("learnlet_fire_n600_shared", "noise"): "FIRE 600, shared",
         ("learnlet_fire_aug", "noise"): "FIRE aug, per scale", ("learnlet_fire_aug_shared", "noise"): "FIRE aug, shared",
         ("learnlet", "pdtau"): "Pipeline, ImageNet", ("learnlet_ext32", "pdtau"): "Pipeline, ImageNet ext.",
         ("learnlet_fire_n600", "pdtau"): "Pipeline, FIRE 600", ("learnlet_fire_shared4", "pdtau"): "Pipeline, FIRE shared"}
for k in list(SHORT):
    if k[1] == "noise" and k[0].startswith("learnlet_fire") and (k[0], "learnt") in STYLE:
        SHORT[(k[0], "learnt")] = SHORT[k] + "*"                     # * = its learnt thresholds

# figure panels (families); a key without runs is skipped
FAMILIES = {
    "pipeline": [("learnlet", "pdtau"), ("learnlet_ext32", "pdtau"), ("learnlet_fire_n600", "pdtau"),
                 ("learnlet_fire_shared4", "pdtau")],
    "imagenet": [("starlet", "noise"), ("learnlet", "noise"), ("learnlet_ext", "noise"), ("learnlet", "learnt"),
                 ("starlet", "learnt")],
    "fire": [("starlet", "noise"), ("learnlet", "noise"), ("learnlet_fire", "noise"), ("learnlet_fire_shared4", "noise"),
             ("learnlet_fire_shared6", "noise"), ("learnlet_fire_n600", "noise"), ("learnlet_fire_n600_shared", "noise"),
             ("learnlet_fire_aug", "noise"), ("learnlet_fire_aug_shared", "noise")],
    "learnt": [("starlet", "noise"), ("learnlet", "learnt"), ("starlet", "learnt"), ("learnlet_fire", "learnt"),
               ("learnlet_fire_shared6", "learnt"), ("learnlet_fire_n600", "learnt"),
               ("learnlet_fire_n600_shared", "learnt"), ("learnlet_fire_aug", "learnt"),
               ("learnlet_fire_aug_shared", "learnt")],
}
# dictionaries (dictionaries/<name>.npz) -> short label and colour, for the model figures
MODELS = {"learnlet": ("ImageNet (64 filters, 4 scales)", HUE["imagenet"], "-"),
          "learnlet_fire": ("FIRE 256 px, per scale (16, 4 scales)", HUE["fire400"], "-"),
          "learnlet_fire_shared4": ("FIRE 256 px, shared (16, 4 trained)", HUE["fire400"], "--"),
          "learnlet_fire_shared6": ("FIRE 256 px, shared (16, 6 trained)", HUE["fire400"], ":"),
          "learnlet_fire_n600": ("FIRE 600 px, per scale (16, 7 scales)", HUE["fire600"], "-"),
          "learnlet_fire_n600_shared": ("FIRE 600 px, shared (16, 7 trained)", HUE["fire600"], "--"),
          "learnlet_fire_aug": ("FIRE scale-augmented, per scale", HUE["fireaug"], "-"),
          "learnlet_fire_aug_shared": ("FIRE scale-augmented, shared", HUE["fireaug"], "--")}
# training options of each FIRE model (train_learnlet_fire.py), for eval_learnlet_fire.py --train-args
TRAIN_ARGS = {"learnlet_fire": "", "learnlet_fire_shared4": "--shared --scales 5",
              "learnlet_fire_shared6": "--shared --scales 7", "learnlet_fire_n600": "--scales 8 --thr-units plane",
              "learnlet_fire_n600_shared": "--shared --scales 8 --thr-units plane",
              "learnlet_fire_aug": "--scales 8 --thr-units plane", "learnlet_fire_aug_shared": "--shared --scales 8 --thr-units plane"}


def rc():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 10.5, "xtick.direction": "in", "ytick.direction": "in",
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "lines.linewidth": 2.0,
                         "lines.markersize": 6.5, "legend.frameon": False, "savefig.facecolor": "white"})
