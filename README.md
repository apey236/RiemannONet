# RiemannONet

Contained within this repository is the codebase for RiemannONet, which is built upon the two-step training DeepONet and U-Net with parameter conditioning. Its purpose is to generate the results presented in the article titled "RiemannONets: Interpretable Neural Operators for Riemann Problems."

## Two-step DeepONet
#### - Instructions for Users

To utilize this repository, you must first clone it to your local machine. Ensure that you have the JAX backend and Python 3.11 installed. Once cloned, navigate to the runFolder directory and execute the following command to run the code:
```python
python main.py
```
#### - Configuring Input Parameters
Adjust input parameters using the `inputs.yaml` file. For LPR and IPR problems, utilize the cos base function of the Rowdy activation function for both trunk and branch networks to replicate the results. For HPR problems, set the trunk network to use tanh and the branch network to use cos. 

## U-Net with parameter conditioning

The parameter-conditioned U-Net developed in TensorFlow 2 can be found in "src_unet" directory. 

**Step 1:** Install the required python packages by executing:
```python
pip install -r src_unet/requirements.txt
```
**Step 2:** Navigate to "src_unet/lp" for the low pressure Sod problem or "src_unet/ip" for the medium pressure Sod problem or "src_unet/hp" for the LeBlanc problem, and train the U-Net with parameter conditioning by executing:
```python
python3 -u train_model.py
```

## Referencing

If you utilize RiemannONet in your research or draft a paper based on results obtained with RiemannONet's assistance, please cite the following publication:

```bibtex
@article{PEYVAN2024116996,
	author = {Ahmad Peyvan and Vivek Oommen and Ameya D. Jagtap and George Em Karniadakis},
	journal = {Computer Methods in Applied Mechanics and Engineering},
	pages = {116996},
	title = {RiemannONets: Interpretable neural operators for Riemann problems},
	volume = {426},
	year = {2024}}
```
## Authors
RiemannONet is written by Ahmad Peyvan and Vivek Oommen (Brown University).

