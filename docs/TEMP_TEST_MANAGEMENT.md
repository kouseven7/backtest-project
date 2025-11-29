# 一時テスト管理システム - 使用方法

## 概要

copilot-instructions.md準拠の一時テスト管理システム。成功した一時テストを自動削除し、履歴を記録します。

## フォルダ構造

```
my_backtest_project/
├── tests/
│   ├── core/              # 継続的なテスト
│   ├── integration/       # 統合テスト
│   ├── temp/              # 一時テスト（成功後削除）
│   │   ├── .gitignore
│   │   └── README.md
│   └── cleanup_temp_tests.py  # 削除スクリプト
├── docs/
│   └── test_history/      # 削除済みテストの履歴
│       ├── README.md
│       └── YYYY-MM.md     # 月次履歴
└── .gitignore             # temp/フォルダを除外
```

## 基本ワークフロー

### 1. 一時テストの作成

```powershell
# tests/temp/にテストファイルを作成
# 命名規則: test_YYYYMMDD_feature_name.py
```

例:
```python
# tests/temp/test_20251128_new_feature.py
"""新機能の動作確認テスト"""

import sys
from pathlib import Path
import importlib.util

# 削除提案機能のインポート（オプション）
project_root = Path(__file__).parent.parent.parent
spec = importlib.util.spec_from_file_location(
    "cleanup_temp_tests",
    project_root / "tests" / "cleanup_temp_tests.py"
)
cleanup_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleanup_module)
suggest_cleanup_after_test = cleanup_module.suggest_cleanup_after_test

def test_feature():
    # 実データでテスト
    assert actual_data > 0
    print("テスト成功")
    return True

if __name__ == "__main__":
    result = test_feature()
    if result:
        # 削除提案を表示
        suggest_cleanup_after_test(__file__)
```

### 2. テスト実行

```powershell
python tests/temp/test_20251128_new_feature.py
```

### 3. 成功確認

削除基準をすべて満たすことを確認:
- [ ] すべてのアサーションが成功
- [ ] 実データでの検証完了（モック/ダミーデータ不使用）
- [ ] フォールバック機能なしで動作確認
- [ ] copilot-instructions.md準拠の検証完了

### 4. テスト削除

#### 方法A: 対話モード（推奨）

```powershell
# 対話的に削除
python tests/cleanup_temp_tests.py
```

処理フロー:
1. 一時テストファイル一覧表示
2. 各ファイルについて削除確認
3. テスト目的の入力（任意）
4. 履歴記録 + 削除実行

#### 方法B: ドライラン（確認のみ）

```powershell
# 削除せずに確認のみ
python tests/cleanup_temp_tests.py --dry-run
```

#### 方法C: 自動削除（上級者向け）

```powershell
# パターン指定で自動削除（要注意！）
python tests/cleanup_temp_tests.py --auto --force --pattern "20251128"
```

警告: `--auto --force` は確認なしで削除します。慎重に使用してください。

## コマンドオプション

### cleanup_temp_tests.py

| オプション | 説明 |
|-----------|------|
| `--dry-run` | ドライラン（削除せず確認のみ） |
| `--auto` | 自動削除モード（対話なし） |
| `--force` | 確認なしで削除（--autoと併用） |
| `--pattern <pattern>` | ファイル名パターン指定 |

### 使用例

```powershell
# 対話モードでドライラン
python tests/cleanup_temp_tests.py --dry-run

# 特定パターンのファイルを対話的に削除
python tests/cleanup_temp_tests.py --pattern "dssms"

# 2025年11月28日のテストを自動削除（危険！）
python tests/cleanup_temp_tests.py --auto --force --pattern "20251128"
```

## テスト履歴の確認

削除されたテストの情報は `docs/test_history/` に記録されます。

```powershell
# 今月の履歴を確認
cat docs/test_history/2025-11.md
```

履歴には以下が記録されます:
- テストファイル名
- 実行日時
- テスト目的
- 検証項目
- 実行結果
- 削除日時
- ファイル情報（サイズ、行数）

## 繰り返し使用するテストの移動

一時テストが繰り返し使用される場合:

```powershell
# 回帰テストとして保存
Move-Item tests/temp/test_feature.py tests/core/

# 統合テストとして保存
Move-Item tests/temp/test_integration.py tests/integration/
```

## Git管理

### 除外対象
- `tests/temp/*.py` - 一時テストファイル
- `tests/temp/*.log` - ログファイル
- `tests/temp/*.json` - 結果ファイル

### 管理対象
- `tests/temp/README.md` - 説明ファイル
- `tests/temp/.gitignore` - gitignore設定
- `docs/test_history/*.md` - テスト履歴

## トラブルシューティング

### Q: 削除したテストを復元できますか？
A: できません。削除前に必ず成功を確認してください。ただし、テスト内容の記録は `docs/test_history/` に残ります。

### Q: 誤って削除した場合は？
A: Gitで管理されていない場合、復元は困難です。重要なテストは `tests/core/` に移動してください。

### Q: テストが失敗した場合は？
A: 削除せず、`tests/temp/` に残してください。問題解決後に再度実行してから削除を検討してください。

### Q: 削除スクリプトが動作しない
A: 以下を確認してください:
1. `tests/temp/` フォルダが存在するか
2. Pythonのバージョン（3.6以上推奨）
3. エラーメッセージの内容

## ベストプラクティス

1. **命名規則の遵守**
   - `test_YYYYMMDD_feature_purpose.py` 形式を推奨
   - 日付を含めることで作成時期が明確

2. **削除提案機能の活用**
   - テストスクリプトに `suggest_cleanup_after_test(__file__)` を追加
   - 成功時に自動的に削除方法が表示される
   - 削除し忘れを防止

3. **削除前の確認**
   - 必ず `--dry-run` で確認してから削除
   - 不明な場合は削除せず保留

4. **履歴記録の活用**
   - テスト目的を簡潔に記入
   - 後から参照できる記録を残す

5. **適切な分類**
   - 一度きりの確認 → `tests/temp/`
   - 回帰テスト → `tests/core/`
   - 統合テスト → `tests/integration/`

6. **定期的な整理**
   - 週次または月次でtemp/フォルダを確認
   - 不要なテストは早めに削除

## サポート

問題が発生した場合:
1. `docs/test_history/` で履歴を確認
2. エラーログを確認
3. copilot-instructions.mdの基本原則を再確認
