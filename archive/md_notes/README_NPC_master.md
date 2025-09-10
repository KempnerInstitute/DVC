# NPC
Statistical method to fit copula distribution from the data in a non parametric way

### Files

- main_all.py                       --> It's the main python file you can use to check with breakpoints
- main_all.ipynb                    --> It's the main jupyter notebook file. You can play with this for plotting also.
- simulations_MI.ipynb              --> Simulations of MI for Gaussian copula with different level of correlations.
- simulations_3vine.ipynb           --> Simulations of vine with dim 3.
- simulations_3vine_binning.ipynb   --> Simulations of vine with dim 3 (generated using bins).
- test_houman_comparison.ipynb      --> Last test I performed on the suggeted data.

### Prerequisites

The program requires

```
python 3.6.0
tensorflow 2.0
```

### Installing

1) Create a virtual environment where to install your packages

```
conda create --name NAME-ENV python=3.6

or (if you do not have anaconda)

virtualenv NAME-ENV

```

2) Activate the environment

```
conda activate --NAME-ENV

or (if you do not have anaconda)

source NAME-ENV/bin/activate
```

3) Install the required packages in the virtual environment (Be sure it is activated)

```
module load gcc/6.2.0 cuda/10.0 python/3.6.0

conda (or pip) install numpy scipy matplotlib ipython jupyter pandas sympy nose
pip install sklearn ipython-autotime

CPU and GPU version:
pip install tensorflow

pip install tensorflow-probability
```
### Run jupyter notebook on local pc

1) If it is not, activate your environment

```
conda activate --NAME-ENV
```
2) Go to your directory

3) Run jupyter notebook

```
jupyter notebook
```

### Run jupyter notebook on O2

1) Open Jupyter notebook, it will show you a link but follow step 2 before using it.

```
ssh -XYC -L 50065:127.0.0.1:50065 -o ServerAliveInterval=60 YOUR-USER@o2.hms.harvard.edu
<<<<<<< HEAD

srun -p gpu_harvey --pty --gres=gpu:1 --account=harvey_contrib -t 1:00:00 bash
=======
>>>>>>> f7b8a763ee33b8502ef221132c35093f4d205575

module load gcc/6.2.0 cuda/10.0 python/3.6.0
export XDG_RUNTIME_DIR=""
source activate YOUR-ENV

export XDG_RUNTIME_DIR=""

jupyter notebook --port=50065 --browser='none'
```

2) Open a new terminal, get access to the same login and to the same node.

```
ssh -o ServerAliveInterval=60 YOUR-USER@login03.o2.rc.hms.harvard.edu  (change login number)

ssh -N -L 50065:127.0.0.1:50065 YOUR-USER@compute-g-16-176  (Select same port and same node)
```

3) Copy the link from the previous terminal and paste it on your default browser.

Now, you're able to run the code.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
