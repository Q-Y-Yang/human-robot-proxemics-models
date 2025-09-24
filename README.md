# Learnable Human-Robot Proxemics Models

This is the code repository for the paper [Learning Human-Robot Proxemics Models from Experimental Data](https://www.mdpi.com/2079-9292/14/18/3704). 


## Main Dependencies
* [mvem](https://mvem.readthedocs.io/en/latest/index.html)
* Python 3.10.18


## Learning the Models
### Proxemic Model
1. Preprocessing the CongreG8 dataset using our scripts, or preprocessing your custom data.
```
python data_utils/data_processing_script.py
```
This script uses two helper scripts: `new_extract_chest_data.py` and `automate_normalized_chest_data.py`.  

- `new_extract_chest_data.py` extracts chest data (position and rotation) from the human dataset and saves it to the folder `all_chest_data` (~140 MB).  
- `automate_normalized_chest_data.py` calculates the relative position and orientation data and saves it to the folder `all_rel_chest_data` (~755 MB).  

Finally, the folder structure should look like this:
```
data_utils/
├── all_chest_data/
└── all_rel_chest_data/
```


2. Learning the parameters of the bivariate skew normal distrubution using `mvem` library (Expectation Maximization):
```
learn_proxemic_model.ipynb
```

### Interaction-position Model
1. Preprocessing the CongreG8 dataset using our scripts, or preprocessing your custom data (the same as Step 1 of Proxemic Model).

2. Learning a KDE model for interaction positions and save the model"
```
learn_interaction_points.ipynb
```

## Applying the Models
* `proxemic_infer_vis.ipynb`: Applying the learned proxemic model on test group and visualizing.
* `interaction_infer_vis.py`: Applying the learned interaction-position model on test data and plotting.

## Evaluation
* `evaluation_proxemic.py`: quantitative evaluation for the proxemic model.
* `evaluation_interaction_kde.py`: quantitative evaluation for the interaction model.
* `evaluation_asym_gaus.py`: quantitative evaluation for a baseline model.

## Deployment




## Citation

If you use the code or the learned models from this repository, please cite
```
@Article{human-robotproxemics,
AUTHOR = {Yang, Qiaoyue and Kachel, Lukas and Jung, Magnus and Al-Hamadi, Ayoub and Wachsmuth, Sven},
TITLE = {Learning Human–Robot Proxemics Models from Experimental Data},
JOURNAL = {Electronics},
VOLUME = {14},
YEAR = {2025},
NUMBER = {18},
ARTICLE-NUMBER = {3704},
URL = {https://www.mdpi.com/2079-9292/14/18/3704},
ISSN = {2079-9292},
DOI = {10.3390/electronics14183704}
}
