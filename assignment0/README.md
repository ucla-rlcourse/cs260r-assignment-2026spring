# Assignment 0: Jupyter Notebook usage, assignment submission workflow and basic pytorch

Hi all! Welcome to our RL course!

This assignment illustrates how to use Jupyter Notebook, the basic of Pytorch and walks you through the assignment submission process!


<a target="_blank" href="https://colab.research.google.com/github/ucla-rlcourse/cs260r-assignment-2026spring/blob/main/assignment0/assignment0.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


## Environment setup instruction

In this course, we require you to have basic knowledge of python.
 
In each assignment, some useful packages will be used to help you. 
For example the reinforcement learning environment [Gym](https://gym.openai.com/), scientific computing [Numpy](https://numpy.org/), machine learning framework [PyTorch](https://pytorch.org/) etc.
We will list the packages required at each assignment. 

In this assignment, you only need to set up your python environment. 

We highly recommend you to set up a conda virtual environment for this course. Here is the advantages to do so:

1. The python packages installed in this course will not affect other projects on your computers since the developing environment is independent to other projects.
2. Other members can run your codes in this course seamlessly, since we are all using the same environment and packages if you set up them following instructions.
3. The robustness and compatibility of codes is also an important criterion to assess your completion of assignments. This is because if the program is not runnable at TA/reader's computer, your submission is considered as failed.
4. A clear and ordered code management habit and also the reproducibility of your code is a good practice for your future research and development work.

First, download the package and install anaconda following the instruction at https://www.anaconda.com/download/success

Then create a new conda environment via typing the line in your console:

```
conda create -n cs260r python=3.11 -y
```

By doing this, you created an environment named `cs260r ` with python 3.11 installed. 
Then you need to activate your environment before doing anything:

```
conda activate cs260r
```

If you activate the environment successfully, you will see `(cs260r) COMPUTERNAME:~ USERNAME$` at your shell.

Then you can install the packages we listed at each assignment like:

```
pip install XXX==1.0.0

# For example, the next command will install torch and torchvision in your virtual environment
pip install torch torchvision

# Install jupytyer notebook
pip install notebook
```

where the `XXX==1.0.0` means to install package `XXX` with the specified version `1.0.0`. 

If you use other packages that you think helpful, you need to list them with the version number at your report of each assignment. Make sure the extra package DO NOT help you to finish the essential part of the assignment. The following example is NOT acceptable.

```python
import numpy as np
from kl_divergence_package_wrote_by_smart_guys import get_kl

def compute_kl(dist1, dist2):
    """
    TODO: You need to implement the computing of KL
    Divergence given two distribution instances.
    
    You should only use numpy package.
    
    The return should be a float that greater than 0.
    """
    return get_kl(dist1, dist2)
```


## Install and use Jupyter notebook

In some assignments, we only provide you with a single jupyter notebook file. 
To open and edit the notebook, you have to install the package first as follows:

```
conda activate cs260r
pip install notebook
```

Now you have installed the jupyter notebook. Go to the directory such as `assignment0`, type the following code in your terminal:

```
jupyter notebook
```

Now you should have opened a jupyter notebook session in your computer. 
Open your browser and go to `http://localhost:8888`  (8888 is the port number, you can change it by starting jupyter notebook via `jupyter notebook --port 8889`).
You will see a beautiful interface provided by jupyter notebook. Now click into `assignmentX.ipynb` and start coding!

For more information on jupyter notebook, please visit: https://jupyter.org/install.html

**Now, please go through the `assignment0.ipynb`.**

