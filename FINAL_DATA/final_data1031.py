import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from transformers import (
    PatchTSMixerConfig,
    PatchTSMixerForPrediction,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from tsfm_public.toolkit.dataset import ForecastDFDataset
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
import os

# 設置參數
file_path = 'content/final_merged_with_lookforward.csv'
predict_columns = ["estOFFRTG", "OFFRTG", "estDEFRTG", "DEFRTG", "estNETRTG", "NETRTG", "ASTpct", "ASTtoTOV", "ASTratio",
                   "ORBpct", "DRBpct", "REBpct", "TOVratio", "effFGpct", "TSpct", "USGpct", "estUSGpct", "estpace", 
                   "pace", "paceper40", "POS", "pie", "FGM", "FGA", "FGpct", "3PM", "3PA", "3Ppct", "FTM", "FTA", 
                   "FTpct", "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS", "plusminusPTS", "SEC", 
                   "PTSoffTOV", "2ndPTS", "FBPTS", "PIP", "oppPTSoffTOV", "opp2ndPTS", "oppFBPTS", "oppPIP", "BLKA", 
                   "foulsdrawn", "SPD", "DIST", "REBchancesOFF", "REBchancesDEF", "REBchancestotal", "touches", 
                   "2ndAST", "FTAST", "passes", "contestedFGM", "contestedFGA", "contestedFGpct", "uncontestedFGM", 
                   "uncontestedFGA", "uncontestedFGpct", "defendedatrimFGM", "defendedatrimFGA", "defendedatrimFGpct"]
retain_columns = ["date", "playerid", "gameid", "home_basic", "home", "win", "team", "name", "date_num"]
context_length = 10
forecast_horizon = 1
batch_size = 32
patch_length = 2
output_file = 'content/nba_predictions2.csv'

output_dir="./output"

# 1. 讀取數據並準備特徵
nba_data = pd.read_csv(file_path)
feature_columns = [col for col in nba_data.columns if col in predict_columns or col in retain_columns]

# 初始化標準化器並進行標準化
scaler = StandardScaler()
nba_data_features_scaled = scaler.fit_transform(nba_data[predict_columns])
nba_data[predict_columns] = nba_data_features_scaled

# 用於存儲所有球員的預測結果
all_predictions = []

# 對每個 player_id 分組進行預測
for player_id, player_data in nba_data.groupby('playerid'):
    player_data = player_data.sort_values('date_num').reset_index(drop=True)

    # 檢查資料長度，確保足夠進行預測
    if len(player_data) <= context_length:
        print(f"Skipping player_id {player_id} due to insufficient data.")
        continue
    
    tsp = TimeSeriesPreprocessor(
        timestamp_column='date_num',
        id_columns=['playerid'],
        target_columns=predict_columns,
        scaling=True,
    )
    tsp.train(player_data)

    train_dataset = ForecastDFDataset(
        tsp.preprocess(player_data),
        id_columns=['playerid'],
        target_columns=predict_columns,
        context_length=context_length,
        prediction_length=forecast_horizon,
    )

    # 設置 PatchTSMixer 模型配置
    config = PatchTSMixerConfig(
        context_length=context_length,
        prediction_length=forecast_horizon,
        patch_length=patch_length,
        num_input_channels=len(predict_columns),
        patch_stride=patch_length,
        d_model=16,
        num_layers=3,
        expansion_factor=2,
        dropout=0.1,
        head_dropout=0.1,
        mode="common_channel",
        scaling="std",
    )
    model = PatchTSMixerForPrediction(config=config)

    # 訓練參數和 Early Stopping 設置
    training_args = TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    early_stopping = EarlyStoppingCallback(early_stopping_patience=3)

    # 訓練模型
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        callbacks=[early_stopping]
    )

    # 進行訓練
    trainer.train()

    # 進行預測
    predictions_tuple = trainer.predict(train_dataset)
    predictions = predictions_tuple.predictions[0]
    if isinstance(predictions, np.ndarray) and predictions.ndim == 3:
        predictions = predictions[:, -1, :]  # 取每個樣本的最後一個時間步長的預測

    # 還原標準化
    predictions = scaler.inverse_transform(predictions)
    predictions_df = pd.DataFrame(predictions, columns=predict_columns)

    # 填充前 10 行為 0，並將 retain_columns 加回
    padding_df = pd.DataFrame(0, index=np.arange(context_length), columns=predict_columns)
    combined_df = pd.concat([padding_df, predictions_df], ignore_index=True)
    combined_df[retain_columns] = player_data[retain_columns]

    all_predictions.append(combined_df)

# 合併所有球員的預測結果並保存至 CSV
final_predictions = pd.concat(all_predictions, ignore_index=True)
final_predictions = final_predictions[predict_columns + retain_columns]  # 保留預測欄位和固定欄位
final_predictions.to_csv(output_file, index=False)

print(f"預測結果已儲存至 {output_file}")
