# ディレクトリ構成

```
pointprocess_crypto
  |-checkpoints
  |-dataframes
  |-images
  |-result
  |-utils
  |  |-datasets.py
  |  |-losses.py
  |-models.py
  |-plot_train_history.py
  |-stochastic_declustering.py
  |-test_pointmodels.py
  |-train.py
```

# utils/datasets.py

暗号資産取引所bybitから指定した期間の約定データをダウンロードし，点過程データに加工してcsvファイルをdataframesフォルダに保存する．

- start_dates：開始日時．複数指定可能．
- end_dates：終了日時．複数指定可能．
- symbols：通貨ペア．複数指定可能．
- v_name：約定データのうち，出来高を表す列名．デフォルトは"size"．
- v_threshold：点過程データにするために約定データを抽出する際，閾値とする出来高．

## 実行例

```
python utils/datasets.py --start_dates 2022/9/1 2022/11/15 2022/12/1 2022/12/8 2022/12/15 --end_dates 2022/11/14 2022/11/30 2022/12/7 2022/12/14 2022/12/21 --symbols BTCUSD ETHUSD XRPUSD --v_name size --v_thresholds 10000 10000 10000
```

[2022/9/1, 2022/11/14]，[2022/11/15, 2022/11/30]，[2022/12/1, 2022/12/7]，[2022/12/8, 2022/12/14]，[2022/12/15, 2022/12/21]の期間の，[BTCUSD，ETHUSD，XRPUSD]の約定データをダウンロードし，dataframesフォルダに保存する．例えば，dataframes/pointprocess_BTCUSD_2022-09-01_2022-11-14.csvは，次のようなデータフレームである．

![](images/pointprocess_df_example.png)

# stochastic_declustering.py

式(7)で表される自己励起点過程のノンパラメトリック推定を行う．bybitから指定した期間の点過程データを取得し，推定された$\mu,g$とL2誤差の履歴$P$をcheckpoints/mu.npy，checkpoints/g.npy，checkpoints/P.npyとして保存し，$\mu,g,\lambda$のグラフと$P$のグラフをimages/mug_estimate.pdf，images/p_err.pdfとして保存する．

- start_date：開始日時
- end_dates：終了日時
- symbols：通貨ペア
- v_name：約定データのうち，出来高を表す列名．デフォルトは"size"．
- v_threshold：点過程データにするために約定データを抽出する際，閾値とする出来高．

## 実行例

```
python stochastic_declustering.py --start_date 2022/3/1 --end_date 2022/3/3 --symbol BTCUSD --v_name size --v_threshold 150000
```