# Usage

## utils/datasets.py

```
python utils/datasets.py --start_dates 2022/9/1 2022/11/15 2022/12/1 --end_dates 2022/11/14 2022/11/30 2022/12/7 --symbols BTCUSD ETHUSD XRPUSD --v_name size --v_thresholds 10000 10000 10000
```

## stochastic_declustering.py

```
python stochastic_declustering.py --start_date 2022/3/1 --end_date 2022/3/3 --symbol BTCUSD --v_name size --v_threshold 150000
```

## train.py

```
python train.py - -model PointFormer - -load_name pointformer_last.tar - -best_save_name pointformer_best.tar - -last_save_name pointformer_last.tar --start_dates 2022/9/1 2022/11/15 - -end_dates 2022/11/14 2022/11/30 - -symbols BTCUSD ETHUSD XRPUSD
```

## plot_train_history.py

```
python plot_train_history.py --load_names pointformer_last.tar pointrnn_last.tar --model_names pointformer pointrnn 
```

## test_pointmodels.py

```
python test_pointmodels.py --model PointFormer --load_name pointformer_best.tar --start_dates 2022/12/1 --end_dates 2022/12/7 --symbols BTCUSD ETHUSD XRPUSD --total_loss_save_name pointformer_total_loss --loss_list_save_name pointformer_loss_list --start_datetime 2022/12/2/00:00:00 --sub_start_datetime 2022/12/2/13:00:00 --sub_end_datetime 2022/12/2/15:00:00 --pred_times_name pointformer_pred_times.pdf
```
