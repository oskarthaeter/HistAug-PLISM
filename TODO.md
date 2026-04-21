# TODO list for HistAug-PLISM


- [x] How to encode scanner/staining information? One-hot or learned embedding?

- [x] Only do same staining transfer, i.e. only transfer between scanners, not between stainings.

- [x] Use more realistic augmentations, like only flips, blur and rotation

- [x] Use organ type tile information to do 80/20 split, i.e. train on some organs and test on others. See plism_organ_loc.csv for organ type information.

- [x] Pre-extract H0-mini features for faster training and testing. See `/mnt/data/plismbench/features/h0_mini/`

- [x] Look at SCORPION [dataset](https://arxiv.org/abs/2507.20907)

