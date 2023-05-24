for i in 10 11 18 19 20
do
    nohup python train.py --c 1.5 --l $i &
    nohup python train.py --c 1.0 --l $i &
    nohup python train.py --c 0.5 --l $i &
done
