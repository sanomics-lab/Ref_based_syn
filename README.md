# Ref_based_syn

## Data
You can download Zinc fragments from this [link](https://drive.google.com/file/d/1DW926e9Xyyg2ggYYJzsLhqMzhBAlyyyp/view?usp=drive_link)

## Installation
```
git https://github.com/sanomics-lab/Ref_based_syn.git
cd Ref_based_syn
conda env create -f environment.yaml
conda activate ref_syn
```
## Data Preparation 
```
python get_embedding.py
python build_dataset.py
python get_embedding.py --input data/matched_bbs.txt --output data/matched_bbs_emb_256.npy
```

## Generation
```
python main.py
```

