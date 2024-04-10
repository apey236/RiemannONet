# RiemannONet

Contained within this repository is the codebase for RiemannONet, which is built upon the two-step training DeepONet framework. Its purpose is to generate the results presented in the article titled "RiemannONets: Interpretable Neural Operators for Riemann Problems."


## Instructions for Users

To utilize this repository, you must first clone it to your local machine. Ensure that you have the JAX backend and Python 3.11 installed. Once cloned, navigate to the runFolder directory and execute the following command to run the code:
```python
python main.py
```
### Configuring Input Parameters
Adjust input parameters using the `inputs.yaml` file. For LPR and IPR problems, utilize the cos base function of the Rowdy activation function for both trunk and branch networks to replicate the results. For HPR problems, set the trunk network to use tanh and the branch network to use cos. 

## Referencing

If you utilize RiemannONet in your research or draft a paper based on results obtained with RiemannONet's assistance, please cite the following publication:

```bibtex
@misc{peyvan2024riemannonets,
      title={RiemannONets: Interpretable Neural Operators for Riemann Problems}, 
      author={Ahmad Peyvan and Vivek Oommen and Ameya D. Jagtap and George Em Karniadakis},
      year={2024},
      eprint={2401.08886},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
## Authors
RiemannONet is written by Ahmad Peyvan (Brown University).

