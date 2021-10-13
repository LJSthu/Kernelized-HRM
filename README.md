# Kernelized-HRM
> Jiashuo Liu, Zheyuan Hu


> The code for our NeurIPS 2021 paper "Kernelized Heterogeneous Risk Minimization"[1]. This repo contains the codes for our **Classification with Spurious Correlation** and **Regression with Selection Bias** simulated experiments, including the data generation process, the whole Kernelized-HRM algorithm and the testing process. 

### Details
There are two files, named `KernelHRM_sim1.py` and `KernelHRM_sim2.py`, which contains the code for the classification simulation experiment and the regression simulation experiment, respectively.
The details of codes are:

* `generate_data_list`: generate data according to the given parameters `args.r_list`.

* `generate_test_data_list`: generate the test data for **Selection Bias** experiment, where the `args.r_list` is pre-defined to [-2.9,-2.7,...,-1.9].

* `main_KernelHRM`: the whole framework for our Kernelized-HRM algorithm. 


### Hypermeters
There are many hyper-parameters to be tuned for the whole framework, which are different among different tasks and require users to carefully tune. Note that although we provide the hyper-parameters for the simulated experiments, it is possible that the results are not exactly the same as ours, which may due to the randomness or something else.  

Generally, the following hyper-parameters need carefully tuned:

* k: controls the dimension of reduced neural tangent features
* whole_epoch: controls the overall number of iterations between the frontend and the backend
* epochs: controls the number of epochs of optimizing the invariant learning module in each iteration
* IRM_lam: controls the strength of the regularizer for the invariant learning
* lr: learning rate
* cluster_num: controls the number of clusters

Further, for the experimental settings, the following parameters need to be specified:

* r_list: controls the strength of spurious correlations
* scramble: similar to IRM[2], whether to mix the raw features
* num_list: controls the number of data points from each environment

As for the optimal hyper-parameters for our simulation experiments, we put them into the `reproduce.sh` file.


### Others
Similar to HRM[3], we view the proposed Kernelized-HRM as a framework, which converts the non-linear and complicated data into linear and raw feature data by neural tangent kernel and includes the clustering module and the invariant prediction module. In practice, one can replace each model to anything they want with the same effect. 

 Though I hate to mention it, our method has the following shortcomings:

 * Just like the original HRM[3], the convergence of the frontend module cannot be guaranteed, and we notice that there may be some cases the next iteration does not improve the current results or even hurts.
 * Hyper-parameters for different tasks may be quite different and need to be tuned carefully.
 * Whether this algorithm can be extended to more complicated image data, such as PACS, NICO *et al.* remains to be seen.(Maybe later we will have a try?)


 ### Reference
 [1] Jiasuho Liu, Zheyuan Hu, Peng Cui, Bo Li, Zheyan Shen. Kernelized Heterogeneous Risk Minimization. *In NeurIPS 2021*.

 [2] Arjovsky M, Bottou L, Gulrajani I, et al. Invariant risk minimization.

 [3] Jiashuo Liu, Zheyuan Hu, Peng Cui, Bo Li, Zheyan Shen. Heterogeneous Risk Minimziation. *In ICML 2021*.