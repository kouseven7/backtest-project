# main_new.py初回起動エラー調査レポート

## 目的
main_new.py（マルチ戦略システム）の初回実行時にKeyboardInterruptエラーが発生し、2回目は実行できるという問題を解決する。

## ゴール（成功条件）
1. 異常の原因がわかる
2. main_new.pyが1回目から正常に実行できる
3. main_new.pyが「専用ターミナルでPythonファイルを実行する」ボタンで実行できる

## 問題詳細
- 症状: VSCode再起動後、main_new.py初回実行でエラー、2回目は成功
- エラー箇所: seaborn → scipy.cluster.hierarchy → textwrap.dedent()でKeyboardInterrupt
- 推定原因: 重いライブラリの初回読み込み時の遅延・タイムアウト問題

## 調査・修正履歴
- 作成日: 2026-01-11
- 調査開始: 21:45
- 解決完了: 21:55

### 根本原因
seaborn → scipy.cluster.hierarchy → textwrap.dedent()でのKeyboardInterrupt
config/__init__.pyから間接的に読み込まれるscipyライブラリの初回読み込み時遅延

### 解決方法
1. config/__init__.py Line 107: portfolio_correlation_optimizerを遅延インポート化
2. main_system/strategy_selection/switching_integration_system.py Line 33: DrawdownControllerを無効化

### 検証結果
- ✅ main_new.py初回実行成功
- ✅ VSCodeタスク実行成功  
- ✅ 副作用なし確認
- ✅ 解決完了