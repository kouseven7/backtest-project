# マルチ戦略バックテストシステム初心者用操作マニュアル

---

## 1. 簡潔な初心者向け操作マニュアル

### 概要
このマニュアルは、マルチ戦略バックテストシステムの「バックテスト」「ペーパートレード」「リアルタイム実行」「戦略最適化」「パラメータ管理」の基本操作を、初めての方でも迷わず実行できるようにまとめたものです。

### 目次
1. 操作前の準備
2. バックテストの実行
3. ペーパートレードの実行
4. リアルタイムデータ実行
5. 戦略最適化の実行
6. パラメータの採用・削除
7. 結果の確認

---

### 1. 操作前の準備
- **目的**: システムが正しく動くか確認する
- **手順**:
  1. Pythonと必要パッケージの確認  
     ```
     python --version
     pip list
     ```
  2. 設定ファイル・ログディレクトリの存在確認  
     ```
     dir config
     dir logs
     ```

### 2. バックテストの実行
- **目的**: 過去データで戦略を検証する
- **手順**:
  ```
  python main.py
  ```
  - 特定銘柄で実行:  
    ```
    python main.py --ticker AAPL
    ```

### 3. ペーパートレードの実行
- **目的**: 仮想取引で戦略を検証する
- **手順**:
  ```
  python demo_paper_trade_runner.py
  ```

### 4. リアルタイムデータ実行
- **目的**: 実際の市場データで戦略を動かす
- **手順**:
  ```
  python demo_realtime_data_system.py
  ```

### 5. 戦略最適化の実行
- **目的**: 戦略パラメータを自動で最適化する
- **手順**:
  ```
  python optimize_strategy.py
  ```
  - VWAPバウンス戦略など個別最適化:
    ```
    python optimization/optimize_vwap_bounce_strategy.py
    ```

### 6. パラメータの採用・削除
- **目的**: 最適化結果を本番に反映・不要なパラメータを削除
- **採用方法**:
  1. 最適化後のパラメータを確認  
     ```
     python -c "from config.optimized_parameters import OptimizedParameterManager; opm = OptimizedParameterManager(); print(opm.list_optimized_params())"
     ```
  2. 承認して本番反映  
     ```
     python -c "from config.optimized_parameters import OptimizedParameterManager; opm = OptimizedParameterManager(); opm.approve_param('VWAP_Bounce', '20250814')"
     ```
     ※戦略名・日付は適宜変更
- **削除方法**:
  ```
  python -c "from config.optimized_parameters import OptimizedParameterManager; opm = OptimizedParameterManager(); opm.delete_param('VWAP_Bounce', '20250814')"
  ```
  ※またはconfig/optimized_parameters.py内の該当エントリを手動で削除

### 7. 結果の確認
- **目的**: 実行結果やエラーを確認する
- **手順**:
  ```
  type logs/main.log
  dir output
  ```

---

## 2. 詳細な運用マニュアル拡張バージョン

### 3.1 戦略パラメータ管理（詳細）

#### 3.1.1 承認済みパラメータ確認
- **目的**: 現在本番で使われているパラメータを確認する
- **手順**:
  ```powershell
  python -c "from config.optimized_parameters import OptimizedParameterManager; opm = OptimizedParameterManager(); print(opm.list_approved_params())"
  ```

#### 3.1.2 新規パラメータ最適化・承認・採用
- **目的**: 戦略のパラメータを最適化し、本番に反映する
- **手順**:
  1. **最適化の実行**
     ```powershell
     python optimize_strategy.py
     ```
     - 個別戦略の場合:
       ```powershell
       python optimization/optimize_vwap_bounce_strategy.py
       ```
  2. **最適化結果の確認**
     ```powershell
     python -c "from config.optimized_parameters import OptimizedParameterManager; opm = OptimizedParameterManager(); print(opm.list_optimized_params())"
     ```
  3. **パラメータの承認・本番反映**
     ```powershell
     python -c "from config.optimized_parameters import OptimizedParameterManager; opm = OptimizedParameterManager(); opm.approve_param('VWAP_Bounce', '20250814')"
     ```
     - ※戦略名・日付は適宜変更
  4. **承認済みパラメータのバックアップ**
     ```powershell
     Copy-Item config/optimized_parameters.py config/optimized_parameters_backup_$(Get-Date -Format "yyyyMMdd").py
     ```

#### 3.1.3 パラメータの削除
- **目的**: 不要なパラメータを削除し、管理を整理する
- **手順**:
  1. **コマンドで削除**
     ```powershell
     python -c "from config.optimized_parameters import OptimizedParameterManager; opm = OptimizedParameterManager(); opm.delete_param('VWAP_Bounce', '20250814')"
     ```
  2. **手動で削除**
     - config/optimized_parameters.pyをエディタで開き、該当エントリを削除

#### 3.1.4 注意事項
- パラメータの承認・削除は必ずバックアップ後に実施してください
- 承認済み以外のパラメータは本番で利用されません
- 削除後は必ずlist_approved_params()で反映状況を確認してください

---

### 3.2 その他の詳細運用手順
- バックテスト、ペーパートレード、リアルタイム実行、結果確認などは「運用マニュアル」本体（docs/operation_manual.md）を参照してください。

---

### 用語注釈
- **バックテスト**: 過去のデータで戦略を検証すること
- **ペーパートレード**: 仮想取引。実際のお金は動かさずに取引をシミュレーション
- **最適化**: 戦略のパラメータを自動で調整し、最も良い結果を探すこと

---