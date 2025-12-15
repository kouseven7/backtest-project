# J-Quants API連携 - 日経225銘柄取得

**作成日**: 2025-12-15  
**最終更新**: 2025-12-15  
**状態**: 調査・実装中

---

## 目次

1. [目的](#目的)
2. [J-Quants API概要](#j-quants-api概要)
3. [認証フロー](#認証フロー)
4. [上場銘柄一覧API](#上場銘柄一覧api)
5. [日経225銘柄の特定方法](#日経225銘柄の特定方法)
6. [実装計画](#実装計画)
7. [重要な制約](#重要な制約)

---

## 目的

**主目的**: J-Quants APIを利用して日経225の最新の上場銘柄一覧情報を取得する

**背景**:
- 現在のnikkei225_components.jsonは手動更新またはハードコード
- 年1回の日経225構成銘柄見直し（10月頃）に自動対応
- 過去のバックテスト再現のため、時点情報の取得が必要

**期待成果**:
- 定期更新スクリプトでの自動銘柄取得
- 正確な日経225構成銘柄リストの維持
- バックテスト再現性の確保

---

## J-Quants API概要

### API基本情報

**ベースURL**: `https://api.jquants.com/v1/`

**登録状況**: 無料プラン登録済み

**無料プランの制限**:
- レートリミット: あり（具体的な制限は利用状況により調整）
- データ期間: 2017年1月以降（Premiumは2008年5月7日以降）
- IDトークン有効期間: 24時間
- リフレッシュトークン有効期間: 1週間

### 重要な留意点

1. **レートリミット**:
   - 一定の使用量を超えると一時的に制限
   - 無料プランでは制限が厳しい可能性

2. **ページング**:
   - 大容量データは`pagination_key`で分割取得
   - レスポンスに`pagination_key`がある限り継続取得

3. **Gzip圧縮**:
   - APIレスポンスはGzip圧縮
   - 一般的なHTTPクライアントは自動対応

---

## 認証フロー

### 3ステップ認証

```
Step 1: メールアドレス + パスワード
   ↓
   リフレッシュトークン取得（有効期間: 1週間）
   ↓
Step 2: リフレッシュトークン
   ↓
   IDトークン取得（有効期間: 24時間）
   ↓
Step 3: IDトークン
   ↓
   API呼び出し
```

### Step 1: リフレッシュトークン取得

**エンドポイント**: `POST /token/auth_user`

**リクエスト**:
```json
{
  "mailaddress": "<YOUR EMAIL_ADDRESS>",
  "password": "<YOUR PASSWORD>"
}
```

**レスポンス**:
```json
{
  "refreshToken": "<YOUR refreshToken>"
}
```

**Curlサンプル**:
```bash
BODY="{\"mailaddress\":\"<YOUR EMAIL_ADDRESS>\", \"password\":\"<YOUR PASSWORD>\"}" && \
curl -X POST -H "Content-Type: application/json" \
  -d "$BODY" https://api.jquants.com/v1/token/auth_user
```

**Pythonサンプル**:
```python
import requests

url = "https://api.jquants.com/v1/token/auth_user"
data = {
    "mailaddress": "<YOUR EMAIL_ADDRESS>",
    "password": "<YOUR PASSWORD>"
}
response = requests.post(url, json=data)
refresh_token = response.json()["refreshToken"]
```

---

### Step 2: IDトークン取得

**エンドポイント**: `POST /token/auth_refresh`

**パラメータ**:
- `refreshtoken` (必須): リフレッシュトークン

**レスポンス**:
```json
{
  "idToken": "<YOUR idToken>"
}
```

**Curlサンプル**:
```bash
REFRESH_TOKEN=<YOUR REFRESH_TOKEN> && \
curl -X POST https://api.jquants.com/v1/token/auth_refresh?refreshtoken=$REFRESH_TOKEN
```

**Pythonサンプル**:
```python
import requests

url = "https://api.jquants.com/v1/token/auth_refresh"
params = {"refreshtoken": refresh_token}
response = requests.post(url, params=params)
id_token = response.json()["idToken"]
```

---

### Step 3: API呼び出し

**認証ヘッダー**:
```
Authorization: Bearer <YOUR idToken>
```

---

## 上場銘柄一覧API

### エンドポイント

`GET /listed/info`

### パラメータ

| パラメータ | 必須 | 説明 |
|----------|------|------|
| code | × | 銘柄コード（指定時は当該銘柄の情報のみ） |
| date | × | 日付（YYYY-MM-DD形式） |

### パラメータ組み合わせとレスポンス

| code指定 | date指定 | レスポンス |
|---------|---------|-----------|
| ○ | ○ | 指定日時点の当該銘柄情報 |
| ○ | × | 直近営業日の当該銘柄情報 |
| × | ○ | 指定日時点の全銘柄情報 |
| × | × | 直近営業日の全銘柄情報 |

### レスポンス例

```json
{
  "info": [
    {
      "Date": "2022-11-11",
      "Code": "86970",
      "CompanyName": "日本取引所グループ",
      "CompanyNameEnglish": "Japan Exchange Group,Inc.",
      "Sector17Code": "16",
      "Sector17CodeName": "金融（除く銀行）",
      "Sector33Code": "7200",
      "Sector33CodeName": "その他金融業",
      "ScaleCategory": "TOPIX Large70",
      "MarketCode": "0111",
      "MarketCodeName": "プライム",
      "MarginCode": "1",
      "MarginCodeName": "信用"
    }
  ]
}
```

### 重要なフィールド

- **Code**: 銘柄コード（5桁）
- **CompanyName**: 会社名
- **ScaleCategory**: 規模区分（**日経225判定に使用**）
- **MarketCode**: 市場コード
- **MarketCodeName**: 市場名
- **Sector17Code/Sector33Code**: 業種コード

---

## 日経225銘柄の特定方法

### 重要な発見

**J-Quants APIのScaleCategoryフィールドで日経225を特定できる可能性が高い**

### ScaleCategoryの種類

公式ドキュメントから推測される分類：
- TOPIX Large70
- TOPIX Mid400
- TOPIX Small
- その他（詳細は要確認）

### 日経225特定の課題

**問題**: J-Quants APIドキュメントに「日経225」を直接示すフラグやフィールドが明記されていない

**調査が必要な点**:
1. `ScaleCategory`に"Nikkei225"や"日経225"などの値が存在するか？
2. 別のフィールドで日経225を判定できるか？
3. J-Quants APIに日経225専用のエンドポイントは存在するか？

### 調査アプローチ

#### オプション1: 全銘柄取得 → フィルタリング
1. `/listed/info`で全銘柄取得
2. レスポンスデータを分析し、日経225銘柄を特定するフィールドを確認
3. 現在の nikkei225_components.json（225銘柄）と照合

#### オプション2: 日経公式データとのクロスチェック
1. J-Quants APIで取得した全銘柄データ
2. 日経公式サイトの日経225構成銘柄リスト
3. 両者を照合して判定ロジックを確立

#### オプション3: 既知の日経225銘柄で検証
1. 現在のnikkei225_components.json（225銘柄）
2. 各銘柄のJ-Quants APIデータを取得
3. 共通するフィールド値を分析

---

## 実装計画

### Phase 1: 調査・検証（1時間）

#### 1-1: 認証情報の設定（10分）
- [ ] 環境変数またはconfig/.envにメールアドレス・パスワードを設定
- [ ] セキュリティ考慮（.gitignoreに追加）

#### 1-2: 認証テスト（20分）
- [ ] リフレッシュトークン取得のテスト
- [ ] IDトークン取得のテスト
- [ ] トークン有効期限の確認

#### 1-3: 上場銘柄一覧APIテスト（30分）
- [ ] 全銘柄取得（date指定なし、code指定なし）
- [ ] レスポンスデータの構造確認
- [ ] ページングの動作確認

#### 1-4: 日経225判定ロジックの確立（30分）
- [ ] オプション3実行: 既知の日経225銘柄でフィールド分析
- [ ] 判定ロジックの確立
- [ ] 検証（225銘柄すべてが正しく判定されるか）

---

### Phase 2: スクリプト実装（1.5時間）

#### 2-1: 基本クラス実装（30分）
- [ ] `JQuantsAPIClient`クラス作成
- [ ] 認証処理の実装
- [ ] トークンキャッシュ機能

#### 2-2: 銘柄取得ロジック実装（40分）
- [ ] 上場銘柄一覧取得
- [ ] ページング対応
- [ ] 日経225フィルタリング

#### 2-3: エラーハンドリング実装（20分）
- [ ] レートリミット対応
- [ ] 認証エラー対応
- [ ] リトライロジック

---

### Phase 3: 統合テスト（1時間）

#### 3-1: 単体テスト（20分）
- [ ] 認証処理のテスト
- [ ] 銘柄取得のテスト
- [ ] フィルタリングのテスト

#### 3-2: 統合テスト（30分）
- [ ] 実際にJ-Quants APIから225銘柄取得
- [ ] 既存JSONファイルとの比較
- [ ] 差分確認

#### 3-3: ドライランテスト（10分）
- [ ] `--dry-run`モードでの動作確認
- [ ] ログ出力確認

---

## 重要な制約

### copilot-instructions.mdに基づく制約

1. **実データ検証必須**
   - J-Quants APIからの実データ取得を確認
   - 225銘柄すべてが正しく取得されることを検証

2. **モック/ダミー禁止**
   - テスト用のダミーデータは使用しない
   - 実際のJ-Quants APIを使用

3. **フォールバック時のログ必須**
   - J-Quants API取得失敗時は必ずログ記録
   - 手動入力モードへの切り替えを明示

### セキュリティ制約

1. **認証情報の保護**
   - メールアドレス・パスワードは環境変数または.envファイル
   - .gitignoreに必ず追加
   - ハードコード禁止

2. **トークンの安全な管理**
   - リフレッシュトークン・IDトークンをファイルキャッシュする場合は暗号化
   - 権限設定（600）

---

## 次のステップ

### 優先度P1: 調査・検証開始

1. **環境設定**
   - [ ] config/.envファイル作成
   - [ ] J-Quants認証情報設定

2. **調査スクリプト作成**
   - [ ] `scripts/jquants_research.py` 作成
   - [ ] 認証テスト実行
   - [ ] 日経225判定ロジック確立

3. **検証**
   - [ ] 225銘柄すべてが取得できることを確認
   - [ ] 既存JSONファイルとの照合

---

## 参考情報

### J-Quants APIドキュメント

- [注意事項](https://jpx.gitbook.io/j-quants-ja/api-reference/attention)
- [リフレッシュトークン取得](https://jpx.gitbook.io/j-quants-ja/api-reference/refreshtoken)
- [IDトークン取得](https://jpx.gitbook.io/j-quants-ja/api-reference/idtoken)
- [上場銘柄一覧](https://jpx.gitbook.io/j-quants-ja/api-reference/listed_info)
- [市場コード](https://jpx.gitbook.io/j-quants-ja/api-reference/listed_info/marketcode)
- [17業種コード](https://jpx.gitbook.io/j-quants-ja/api-reference/listed_info/sector17code)
- [33業種コード](https://jpx.gitbook.io/j-quants-ja/api-reference/listed_info/sector33code)

### 関連ファイル

- `config/dssms/nikkei225_components.json` - 既存の銘柄リスト（225銘柄）
- `docs/dssms/nikkei225_management/README.md` - 未完了タスク管理

---

## 更新履歴

| 日付 | 内容 | 担当者 |
|------|------|--------|
| 2025-12-15 | 初版作成、J-Quants API調査完了 | AI Assistant |
