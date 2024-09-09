# iCrossformer

## Usage 

1. Install Pytorch and necessary dependencies.

```
pip install -r requirements.txt
```

1. The datasets can be obtained from [Google Drive](https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view?usp=drive_link) or [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/2ea5ca3d621e4e5ba36a/).

2. Train and evaluate the model. We provide all the above tasks under the folder ./scripts/. You can reproduce the results as the following examples:

```
# Multivariate forecasting with iTransformer
bash ./scripts/multivariate_forecasting/ECL/iCrossformer.sh

# Ablation Study
bash ./scripts/ablation_study/DSW+TSA/Exchange/iCrossformer.sh

# Impact of Segments
bash ./scripts/seg_num/24/ETT/iCrossformer_ETTh2.sh

# Impact of batchsize
bash ./scripts/batchsize/6/SolarEnergy/iCrossformer.sh