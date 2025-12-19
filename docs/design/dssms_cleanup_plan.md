"""
DSSMS設計違反コード削除計画

Phase 1: DSSMS側の削除
======================

【削除対象メソッド】

1. _close_position() (Line 2321-2410)
   - 削除理由: PaperBrokerが担当すべき処理
   - 影響範囲: 3箇所の呼び出し
   - 代替: main_new.pyのForceClose戦略

2. _open_position() (Line 2414-2480)
   - 削除理由: PaperBrokerが担当すべき処理
   - 影響範囲: 2箇所の呼び出し
   - 代替: main_new.pyの各戦略のエントリー判断

3. _calculate_position_update() (Line 2204-2319)
   - 削除理由: 未使用メソッド
   - 影響範囲: なし（呼び出し箇所なし）

【削除対象コード（呼び出し箇所）】

1. Line 1647-1677: 銘柄切替時の_close_position()と_open_position()呼び出し
   ```python
   if self.current_symbol and self.position_size > 0:
       close_result = self._close_position(self.current_symbol, target_date)  # 削除
   open_result = self._open_position(selected_symbol, target_date)  # 削除
   ```

2. Line 505: バックテスト終了時の_close_position()呼び出し
   ```python
   close_result = self._close_position(self.current_symbol, final_trading_date)  # 削除
   ```

3. Line 2289: 通常取引での_close_position()呼び出し（未使用ブロック内）
   ```python
   close_result = self._close_position(symbol, target_date)  # 削除
   ```

4. Line 2250: 通常取引での_open_position()呼び出し（未使用ブロック内）
   ```python
   position_result = self._open_position(symbol, target_date)  # 削除
   ```

【削除対象変数】

1. self.position_size (Line 157初期化 + 30箇所以上の参照)
   - 削除理由: PaperBrokerが管理すべき
   - 代替: PaperBroker.get_positions()から取得

2. self.current_symbol（ポジション管理用）
   - 削除理由: PaperBrokerが管理すべき
   - 代替: PaperBroker.get_positions()から取得
   - 注意: 銘柄選択結果の保持用途は残す

【削除対象ロジック】

1. execution_type='switch'設定 (Line 1660, 1674)
   ```python
   close_result['execution_detail']['execution_type'] = 'switch'  # 削除
   open_result['execution_detail']['execution_type'] = 'switch'  # 削除
   ```

2. switch_execution_details収集 (Line 1643-1678)
   - 削除理由: DSSMSが取引を実行しないため
   - 代替: main_new.pyのexecution_detailsから取得

3. switch_cost計算 (Line 1680-1688)
   - 残す理由: 銘柄切替のコスト計算は必要
   - 修正: PaperBrokerの手数料として処理

【影響を受けるメソッド】

1. _evaluate_and_execute_switch() (Line 1606-1726)
   - 大幅な修正必要
   - 銘柄選択結果のみをmain_new.pyに渡す形に変更

2. run_dynamic_backtest() (Line 398-560)
   - バックテスト終了時のForceClose削除
   - main_new.pyに移管

3. _process_daily_trading() (Line 564-725)
   - position_size参照を削除
   - PaperBroker経由に変更

4. _generate_final_results() (Line 2521-2655)
   - position_size参照を削除
   - PaperBroker経由に変更

【段階的削除計画】

Stage 1: 未使用メソッドの削除
  - _calculate_position_update() 削除
  - 影響なし（呼び出し箇所なし）

Stage 2: 銘柄切替時の取引実行削除
  - Line 1647-1677のコード削除
  - switch_execution_details収集削除
  - execution_type='switch'削除

Stage 3: _close_position()と_open_position()削除
  - メソッド本体削除
  - 残りの呼び出し箇所削除

Stage 4: position_size管理削除
  - self.position_size初期化削除
  - 参照箇所をPaperBroker経由に変更

Stage 5: バックテスト終了処理移管
  - Line 505のForceClose削除
  - main_new.pyに移管
"""
