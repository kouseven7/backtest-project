# 危険なDSSMSファイル無効化ログ

## 無効化されたファイル
- src\dssms\unified_output_engine.py -> unified_output_engine.py.DISABLED_*
- src\dssms\dssms_excel_exporter.py -> dssms_excel_exporter.py.DISABLED_*
- src\dssms\dssms_switch_engine_v2.py -> dssms_switch_engine_v2.py.DISABLED_*
- src\dssms\dssms_switch_coordinator_v2.py -> dssms_switch_coordinator_v2.py.DISABLED_*
- src\dssms\comprehensive_scoring_engine.py -> comprehensive_scoring_engine.py.DISABLED_*

## バックアップ場所
すべてのオリジナルファイルは以下に保存されています：
- backup_dangerous_files_20250930_082625

## プレースホルダーファイル
以下のファイルは最小限の安全実装に置き換えられました：
- src\dssms\unified_output_engine.py
- src\dssms\dssms_excel_exporter.py
- src\dssms\dssms_switch_engine_v2.py
- src\dssms\dssms_switch_coordinator_v2.py
- src\dssms\comprehensive_scoring_engine.py

## 復元方法
復元するには、.DISABLED_*ファイルの拡張子を削除するか、バックアップディレクトリからファイルをコピーしてください。