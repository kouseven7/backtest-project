# 一時テストフォルダ (temp/)

## 目的
一時的な動作確認テストを配置するフォルダ。成功後は自動削除されます。

## 使用方法

### 1. テスト作成
```python
# tests/temp/test_YYYYMMDD_feature_name.py
# 日付を含めた命名規則を推奨
```

### 2. テスト実行
```powershell
python tests/temp/test_YYYYMMDD_feature_name.py
```

### 3. 成功後の処理
```powershell
# 自動削除スクリプトを使用
python tests/cleanup_temp_tests.py
```

## 削除基準

以下のすべてを満たす場合、テストを削除できます:

- [ ] すべてのアサーションが成功
- [ ] 実データでの検証完了（モック/ダミーデータ不使用）
- [ ] フォールバック機能なしで動作確認
- [ ] copilot-instructions.md準拠の検証完了
- [ ] テスト結果がdocs/test_history/に記録済み

## 注意事項

1. **削除禁止ケース**
   - テストが失敗した場合
   - 不明なエラーが発生した場合
   - 繰り返し使用する可能性がある場合

2. **移動推奨ケース**
   - 回帰テストとして使用する場合 → `tests/core/`
   - 統合テストとして使用する場合 → `tests/integration/`

3. **Git管理**
   - このフォルダの内容はGitで追跡されません
   - テスト結果のみdocs/test_history/に記録されます

## ファイル命名規則

推奨形式:
```
test_YYYYMMDD_<feature>_<purpose>.py
```

例:
- `test_20251128_dssms_fallback_removal.py`
- `test_20251128_var_basic_validation.py`
- `test_20251128_portfolio_integration.py`
