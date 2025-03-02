# UniDepth-on-NYUV2
This repository contains a Python script built on top of [UniDepth](https://github.com/lpiccinelli-eth/UniDepth).

Running this repository requires a correctly set-up Unidepth environment.
In order to run the scripts:   
1- Clone the repository [here](https://github.com/lpiccinelli-eth/UniDepth) and create environment / install requirements as described there.  
2- Change the environment name in .sh files (If you went with the suggested name Unidepth, this step is not required).  
3- Run the respective script using:  
```bash
bash v1_test.sh
```  
or   
```bash  
bash v2_test.sh
```

## About the code
Once you run the script, it will try to download the dataset under /home/{your_username}/nyu_cache. All my scripts use the cache there, so if you already have it please move the dataset to there.  
  
It is further possible to change the dataset sampling by:  

```python
dataset = load_dataset("sayakpaul/nyu_depth_v2", split="train[:40000]", cache_dir="/home/") # Loads the dataset
dataset = dataset.select(range(0, 40000, 40))  # Samples every 40th data
```



## Acknowledgment
This work is based on [UniDepth](https://github.com/lpiccinelli-eth/UniDepth), developed by [Luigi Picinelli](https://github.com/lpiccinelli-eth).

