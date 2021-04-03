python CLDR.py --replay 200 --dataset 's-mnist'  --alpha 2  --beta 0.1  --gamma 0.1
python CLDR.py --replay 500 --dataset 's-mnist'  --alpha 2  --beta 0.1  --gamma 0.01
python CLDR.py --replay 200 --dataset 'p-mnist'  --alpha 1  --beta 0.001  --gamma 0.01
python CLDR.py --replay 500 --dataset 'p-mnist'  --alpha 1  --beta 0.001  --gamma 0.01
python CLDR.py --replay 200 --dataset 'cifar10'   --alpha 2  --beta 0.1  --gamma 0.1
python CLDR.py --replay 500 --dataset 'cifar10'   --alpha 2  --beta 0.1  --gamma 0.05
python CLDR.py --replay 200 --dataset 'cifar100'   --alpha 2  --beta 0.1  --gamma 0.0001
python CLDR.py --replay 500 --dataset 'cifar100'   --alpha 2  --beta 0.1  --gamma 0.0001
