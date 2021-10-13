# For the Classification with Spurious Correlation data
# r = 0.7
python3 KernelHRM_sim1.py --method IGD --k 15 --scramble 1 --whole_epoch 19 --device 6 --epochs 2000

# r = 0.75
python3 KernelHRM_sim1.py --method IGD --k 15 --scramble 1 --whole_epoch 10 --device 6 --epochs 3000

# r = 0.8
python3 KernelHRM_sim1.py --method IGD --k 10 --scramble 1 --whole_epoch 5 --device 4 --epochs 1000




# For the Regression with Selection Bias data
# Scenario 1
# r = 1.5
python3 KernelHRM_sim2.py --r_list 1.5 -1.1 -2.5 --num_list 1000 100 1000 --method IGD \
                    --epochs 3000 --lr 7e-3 --k 40 --IRM_lam 0.1 --whole_epoch 20 --scramble 0 --IRM_ann 500

# r = 1.9
python3 KernelHRM_sim2.py --r_list 1.9 -1.1 -2.5 --num_list 1000 100 1000 --method IGD \
                    --epochs 3000 --lr 7e-3 --k 30 --IRM_lam 0.1 --whole_epoch 20 --scramble 0 --IRM_ann 500

# r = 2.3
python3 KernelHRM_sim2.py --r_list 2.3 -1.1 -2.5 --num_list 1000 100 1000 --method IGD \
                    --epochs 3000 --lr 7e-3 --k 20 --IRM_lam 0.1 --whole_epoch 20 --scramble 0 --IRM_ann 500


# Scenario 2
# r = 1.5
python3 KernelHRM_sim2.py --r_list 1.5 -1.1 -2.5 --num_list 1000 100 1000 --method IGD \
                    --epochs 3000 --lr 7e-3 --k 40 --IRM_lam 0.1 --whole_epoch 20 --scramble 1 --IRM_ann 500

# r = 1.9
python3 KernelHRM_sim2.py --r_list 1.9 -1.1 -2.5 --num_list 1000 100 1000 --method IGD \
                    --epochs 3000 --lr 7e-3 --k 30 --IRM_lam 0.1 --whole_epoch 20 --scramble 1 --IRM_ann 500


# r = 2.3
python3 KernelHRM_sim2.py --r_list 2.3 -1.1 -2.5 --num_list 1000 100 1000 --method IGD \
                    --epochs 3000 --lr 7e-3 --k 20 --IRM_lam 0.1 --whole_epoch 20 --scramble 1 --IRM_ann 500