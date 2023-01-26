# RecNet: Early Attention Guided Feature Recovery
***

### Download DCASE2021 Dataset
Download the following files from [here](https://zenodo.org/record/4844825#.Y9KkxtLMKEA).
- metadata_dev.zip
- mic_dev.zip
- mic_dev.z01
- mic_eval.zip
***
### Getting started

- Clone the repository
- Run the following commands
```bash
mkdir result_txt
python3.8 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

***
### Project Structure

***
### Create the full-rank data (Must be done before running RecNet)
```bash
python main.py --full_rank
```
***
### Train the baseline
```bash
python main.py --type_=1 --seed=42 --epochs=100 --pct=0.75
```
Change the ``seed`` and ``pct`` accordingly.
***

### Train the RecNet
```bash
python main.py --type_=2 --seed=42 --epochs=300 --pct=0.75
```
Change the ``seed`` and ``pct`` accordingly.
***

### Test the full-rank model (FR) with low-rank data
```bash
python main.py --test_fr --type_=1 --seed=<seed> --epochs=100 --pct=0.75 --test_model_name="./models/baseline_<seed>_0.h5"
```
Change the ``seed`` and ``pct`` accordingly.
***

### Resuts
Results can be found at ``result_txt`` folder.
***

### Disclaimer
Implementation of baseline is taken form [here](https://github.com/sharathadavanne/seld-dcase2022).
