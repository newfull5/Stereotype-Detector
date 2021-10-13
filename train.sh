python3 train.py --model_name tunib/electra-ko-base \
                 --data_dir ./data \
                 --patience 5 \
                 --gpus [0] \
                 --lr 3e-5 \
                 --eps 1e-8 \
                 --batch_size 8 \
                 --num_workers 16 \
                 --num_labels 7