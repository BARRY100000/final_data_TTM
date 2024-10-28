import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from transformers import (
    PatchTSMixerConfig,
    PatchTSMixerForPrediction,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from tsfm_public.toolkit.dataset import ForecastDFDataset
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor

# 設置參數
file_path = 'content/final_data_4.csv'
label_columns = ['date', 'playerid', 'gameid', 'home_basic', 'home', 'win', 'team', 'name', 'date_num']
context_length = 10
forecast_horizon = 1
batch_size = 32
patch_length = 2
output_file = 'content/nba_predictions.csv'

# 1. 讀取數據並準備特徵
nba_data = pd.read_csv(file_path)
feature_columns = [col for col in nba_data.columns if col not in label_columns]

# 初始化標準化器並進行標準化
scaler = StandardScaler()
nba_data_features_scaled = scaler.fit_transform(nba_data[feature_columns])
nba_data[feature_columns] = nba_data_features_scaled

# 用於存儲所有球員的預測結果和損失分析
all_predictions = []
loss_summary = []

# 對每個 player_id 分組進行預測和分析
for player_id, player_data in nba_data.groupby('playerid'):
    player_data = player_data.sort_values('date_num').reset_index(drop=True)

    # 檢查資料長度，確保足夠進行預測
    if len(player_data) <= context_length:
        print(f"Skipping player_id {player_id} due to insufficient data.")
        continue
    
    tsp = TimeSeriesPreprocessor(
        timestamp_column='date_num',
        id_columns=['playerid'],
        target_columns=feature_columns,
        scaling=True,
    )
    tsp.train(player_data)

    train_dataset = ForecastDFDataset(
        tsp.preprocess(player_data),
        id_columns=['playerid'],
        target_columns=feature_columns,
        context_length=context_length,
        prediction_length=forecast_horizon,
    )

    # 設置 PatchTSMixer 模型配置
    config = PatchTSMixerConfig(
        context_length=context_length,
        prediction_length=forecast_horizon,
        patch_length=patch_length,
        num_input_channels=len(feature_columns),
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

    # 檢查是否存在檢查點
    last_checkpoint = None
    if os.path.isdir(output_dir):
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            last_checkpoint = os.path.join(output_dir, max(checkpoints, key=lambda x: int(x.split("-")[1])))
            print(f"Resuming training from checkpoint: {last_checkpoint}")
        else:
            print("No checkpoints found. Starting training from scratch.")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        callbacks=[early_stopping]
    )

    # 訓練模型
    trainer.train()

    # 進行預測
    predictions_tuple = trainer.predict(train_dataset)
    predictions = predictions_tuple.predictions[0]
    if isinstance(predictions, np.ndarray) and predictions.ndim == 3:
        predictions = predictions[:, -1, :]  # 取每個樣本的最後一個時間步長的預測

    # 還原標準化
    predictions = scaler.inverse_transform(predictions)
    predictions_df = pd.DataFrame(predictions, columns=feature_columns)

    # 填充前 10 行為 0，並將 label_columns 加回
    padding_df = pd.DataFrame(0, index=np.arange(context_length), columns=feature_columns)
    combined_df = pd.concat([padding_df, predictions_df], ignore_index=True)
    combined_df[label_columns] = player_data[label_columns]

    all_predictions.append(combined_df)

    # 計算損失函數
    original_data = scaler.inverse_transform(player_data[feature_columns][context_length:])
    predicted_data = predictions[:len(original_data)]

    mse = mean_squared_error(original_data, predicted_data)
    mae = mean_absolute_error(original_data, predicted_data)
    loss_summary.append([player_id, mse, mae])

    # 儲存該 player_id 的預測結果到單獨的 CSV 文件
    player_results = pd.DataFrame({
        "Original": original_data[:, 0],  # 只取第一個特徵作為範例
        "Predicted": predicted_data[:, 0]
    })
    player_results.to_csv(f'content/player_{player_id}_results.csv', index=False)

# 儲存所有球員的損失總結到 CSV 文件
loss_summary_df = pd.DataFrame(loss_summary, columns=["PlayerID", "MSE", "MAE"])
loss_summary_df.to_csv("content/loss_summary.csv", index=False)

# 合併所有球員的預測結果
final_predictions = pd.concat(all_predictions, ignore_index=True)
final_predictions.to_csv(output_file, index=False)
