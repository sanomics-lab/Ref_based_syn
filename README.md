# Ref_based_syn

Code for the paper "Artificial Intelligence-Assisted Optimization of Antipigmentation Tyrosinase Inhibitors: De Novo Molecular Generation Based on a Low Activity Lead Compound"

## Platform
This research is based on MolProphet: A One-Stop, General Purpose, and AI-Based Platform for the Early Stages of Drug Discovery

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/dw8h0BBQJvY/0.jpg)](https://www.youtube.com/watch?v=dw8h0BBQJvY)

[Website](https://www.molprophet.com) | [Video Introduction](https://www.youtube.com/watch?v=dw8h0BBQJvY) | [Paper](https://doi.org/10.1021/acs.jcim.3c01979)

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

## Citation
Cai, H., Chen, W., Jiang, J., Wen, H., Luo, X., Li, J., Lu, L., Zhao, R., Ni, X., Sun, Y., Wang, J., Li, Z., Ju, B., Jiang, X., & Bai, R. (2024). Artificial Intelligence-Assisted Optimization of Antipigmentation Tyrosinase Inhibitors: De Novo Molecular Generation Based on a Low Activity Lead Compound. In Journal of Medicinal Chemistry. American Chemical Society (ACS). https://doi.org/10.1021/acs.jmedchem.4c00091

## Reference
Yang, K., Xie, Z., Li, Z., Qian, X., Sun, N., He, T., Xu, Z., Jiang, J., Mei, Q., Wang, J., Qu, S., Xu, X., Chen, C., & Ju, B. (2024). MolProphet: A One-Stop, General Purpose, and AI-Based Platform for the Early Stages of Drug Discovery. In Journal of Chemical Information and Modeling (Vol. 64, Issue 8, pp. 2941–2947). American Chemical Society (ACS). https://doi.org/10.1021/acs.jcim.3c01979
