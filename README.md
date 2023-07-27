# CLiST
This is a Pytorch implementation of A Cross-time Linear Spatial-temporal Transformer for Traffic Forecasting (CLiST).

## Data
For short-term forcasting, the three PEMS datasets (PEMS04, PEMS07, and PEMS08) are available at [ASTGNN](https://github.com/guoshnBJTU/ASTGNN) or [STSGCN](https://github.com/Davidham3/STSGCN), and the data files (`PEMS04.npz`, `PEMS07.npz`, and `PEMS08.npz`) should be put into the `./datasets` folder. And the three Grid datasets (NYCTaxi, CHIBike, and T-Drive) are avaliable at [PDFormer](https://github.com/BUAABIGSCity/PDFormer), you may download the raw data files(`NYCTaxi.grid`, `CHIBike.grid`, and `T-Drive.grid`), and then put them into the `./datasets` folder.

For long-term forecasting, two PEMS datasets are available at [SSTBAN](https://github.com/guoshnBJTU/SSTBAN), you may download two files (`pems04_1dim.npz` and `pems08_1dim_npz`) and put them into `./datasets` folder.

## Requirements
Python 3.9.12, torch 1.11.0, numpy 1.21.5, einops 0.6.0

## Usage
The commands on PEMS08 are presented as example.

Step 1: Generate Datasets (e.g., using one-hour history data to predict data in the next hour)
```python
python prepare_datasets.py --config_file PEMS08_12_12.json --save 1
```

Step 2: Train and Test
```python
python trainCLiST.py --config_file PEMS08_12_12.json
```

The hyper-parameter settings for all datasets are given in `.configurations/` folder.