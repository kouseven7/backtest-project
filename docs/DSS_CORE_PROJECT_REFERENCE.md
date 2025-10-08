# DSS Core Project 開発ガイド# DSS Core Project 開発ガイド

**DSS（Dynamic Stock Selector）バックテスター V3 構築リファレンス****DSS（Dynamic Stock Selector）バックテスター V3 構築リファレンス**



## プロジェクト概要## プロジェクト概要



**目標**: 既存DSSMSコンポーネントを統合したDSS銘柄選択エンジン構築  **目標**: DSS基本機能（銘柄選択エンジン）に特化したシンプルなバックテスター作成  

**成果物**: `src/dssms/dssms_backtester_v3.py`  **成果物**: `src/dssms/dssms_backtester_v3.py`  

**設計思想**: 実証済みコンポーネント活用 → 高度な銘柄選択 → 安定動作**設計思想**: 毎日のランキング更新 → パーフェクトオーダー1位銘柄選定 → 結果出力



## 開発方針：既存ファイル統合アプローチ## DSS Core 5段階フロー



### **なぜ既存ファイル統合か**### **1. データを取得する**

[OK] **実証済みのロジック**: 既存ファイルは動作確認済み  **概要**: 複数銘柄の市場データを並列取得

[OK] **設定の活用**: `config/dssms/*.json`の豊富な設定を利用  ```python

[OK] **明確な責任分離**: 各ファイルが独立した機能を持つ  # ハードコード銘柄リスト（シンプル開始）

[OK] **デバッグ容易性**: 問題の所在が特定しやすい  symbol_universe = ['7203', '9984', '6758', '4063', '8306', '6861', '7741', '9432', '8058', '9020']

```

[ERROR] **シンプル実装の問題点**: 機能重複実装・段階的複雑化・テスト範囲拡大**関連ファイル・パス：**

- `src/dssms/dssms_data_manager.py` - DSSMS専用データ管理

---- `config/dssms/nikkei225_components.json` - 日経225銘柄リスト（将来拡張用）

- `data_fetcher.py` (フォールバック用)

## DSS Core 統合コンポーネント

**V3実装ポイント**:

### **1. パーフェクトオーダー検出**- 初期版では10銘柄程度に限定

**ファイルパス**: `src/dssms/perfect_order_detector.py`  - yfinanceによる直接取得（DSSMSDataManagerは後回し）

**機能**: 移動平均配置によるトレンド強度判定  - エラーハンドリングはシンプルに

**設定**: 内部パラメータまたは設定ファイル  

### **2. データを処理する**

**V3での活用**:**概要**: 取得データの品質チェックと前処理

```python**関連ファイル・パス：**

self.perfect_order_detector = PerfectOrderDetector()- `src/dssms/data_quality_enhancer.py` - データ品質向上（オプショナル）

- `src/dssms/data_cleaning_engine.py` - データクリーニング（オプショナル）

# 使用例

perfect_order_results = self.perfect_order_detector.analyze_symbols(**V3実装ポイント**:

    market_data, target_date- 最小限の処理（欠損値チェック程度）

)- 移動平均計算（5日、25日、75日）のみ実装

```

### **3. トレンド判断をする（パーフェクトオーダー）**

### **2. 市場状況監視****概要**: 移動平均の配置によるトレンド強度判定

**ファイルパス**: `src/dssms/market_condition_monitor.py`  **関連ファイル・パス：**

**機能**: 市場全体の状況判定  - `src/dssms/perfect_order_detector.py` - パーフェクトオーダー検出

**活用**: パーフェクトオーダー判定の補完情報  

**V3実装ポイント**:

### **3. 階層的ランキングシステム**```python

**ファイルパス**: `src/dssms/hierarchical_ranking_system.py`  def detect_perfect_order(self, symbol_data: pd.DataFrame) -> float:

**機能**: 複数指標による銘柄順位付け      """

**設定**: `config/dssms/ranking_config.json`      パーフェクトオーダー判定

    Returns: 0.0-1.0 のスコア（1.0が最強の上昇トレンド）

**V3での活用**:    """

```python    # MA5 > MA25 > MA75 and 価格 > MA5 = 完璧なパーフェクトオーダー

self.ranking_system = HierarchicalRankingSystem()    # 段階的にスコア化

```

# 使用例

ranking_results = self.ranking_system.rank_symbols(### **4. ランキングする**

    scoring_results, target_date**概要**: 各銘柄のパーフェクトオーダースコアで順位付け

)**関連ファイル・パス：**

```- `src/dssms/hierarchical_ranking_system.py` - 階層的ランキング（参考のみ）

- `src/dssms/comprehensive_scoring_engine.py` - 総合スコアリング（参考のみ）

### **4. 総合スコアリングエンジン**

**ファイルパス**: `src/dssms/comprehensive_scoring_engine.py`  **V3実装ポイント**:

**機能**: 複数指標を統合したスコア計算  ```python

**活用**: ランキングシステムへの入力データ生成  def rank_symbols(self, symbol_data_dict: Dict[str, pd.DataFrame]) -> List[Dict]:

    """

**V3での活用**:    銘柄ランキング実行

```python    Returns: [{'symbol': '7203', 'score': 0.95, 'rank': 1}, ...]

self.scoring_engine = ComprehensiveScoringEngine()    """

    # シンプルなスコア順ソート

# 使用例```

scoring_results = self.scoring_engine.calculate_comprehensive_scores(

    market_data, perfect_order_results### **5. 上位ランキングの銘柄を選択する**

)**概要**: ランキング1位の銘柄を抽出

```**関連ファイル・パス：**

- `src/dssms/intelligent_switch_manager.py` - インテリジェント切替管理（参考のみ）

### **5. インテリジェント切替管理**

**ファイルパス**: `src/dssms/intelligent_switch_manager.py`  **V3実装ポイント**:

**機能**: 銘柄切替の最終判定  ```python

**設定**: `config/dssms/hierarchical_switch_decision_config.json`  def select_top_symbol(self, ranking_result: List[Dict]) -> str:

    """

**V3での活用**:    最上位銘柄選択

```python    Returns: 銘柄コード（例: '7203'）

self.switch_manager = IntelligentSwitchManager()    """

    return ranking_result[0]['symbol']  # 1位を単純選択

# 使用例```

selection_result = self.switch_manager.evaluate_switch(

    current_symbol=self.current_position,## dssms_backtester_v3.py 実装仕様

    ranking_results=ranking_results,

    market_data=market_data### クラス設計

)```python

```class DSSBacktesterV3:

    """

---    DSS（Dynamic Stock Selector）バックテスター V3

    

## 推奨開発フロー    目標: 銘柄選択エンジンとしての機能に特化

    範囲: データ取得 → パーフェクトオーダー判定 → ランキング → 1位選択

### **Phase 1: 個別ファイル動作テスト**    """

**目標**: 各コンポーネントの独立動作確認      def __init__(self):

**期間**: 1日          self.logger = setup_logger(__name__)

        self.symbol_universe = ['7203', '9984', '6758', '4063', '8306']  # 初期5銘柄

**実行内容**:        

```bash    def run_daily_selection(self, target_date: datetime) -> Dict[str, Any]:

# インポートテスト        """日次銘柄選択実行（メイン処理）"""

python -c "from src.dssms.perfect_order_detector import PerfectOrderDetector; print('Perfect Order: OK')"        

python -c "from src.dssms.hierarchical_ranking_system import HierarchicalRankingSystem; print('Ranking: OK')"    def fetch_market_data(self, symbols: List[str], date: datetime) -> Dict[str, pd.DataFrame]:

python -c "from src.dssms.comprehensive_scoring_engine import ComprehensiveScoringEngine; print('Scoring: OK')"        """市場データ取得"""

python -c "from src.dssms.intelligent_switch_manager import IntelligentSwitchManager; print('Switch Manager: OK')"        

python -c "from src.dssms.market_condition_monitor import MarketConditionMonitor; print('Market Monitor: OK')"    def calculate_perfect_order_scores(self, market_data: Dict) -> Dict[str, float]:

```        """パーフェクトオーダースコア計算"""

        

**チェックリスト**:    def rank_symbols(self, scores: Dict[str, float]) -> List[Dict]:

- [ ] 全ファイルのインポート成功        """銘柄ランキング"""

- [ ] 依存関係エラーなし        

- [ ] 設定ファイル読み込み確認    def select_top_symbol(self, ranking: List[Dict]) -> str:

- [ ] 基本メソッド呼び出し確認        """1位銘柄選択"""

```

### **Phase 2: 統合クラス基本構造作成**

**目標**: DSSBacktesterV3の骨格作成  ### 実装範囲と制限

**期間**: 1日  

[OK] **V3で実装**:

**実装内容**:1. **データ取得**: yfinance直接利用（5-10銘柄）

```python2. **パーフェクトオーダー判定**: MA5>MA25>MA75 判定

class DSSBacktesterV3:3. **ランキング**: スコア順ソート

    """既存DSSMSコンポーネント統合バックテスター"""4. **1位選択**: ランキング[0]の単純選択

    5. **結果出力**: JSON/Dict形式

    def __init__(self):

        self.logger = setup_logger(__name__)[ERROR] **V3で実装しない**:

        - 取引シミュレーション機能

        # 既存コンポーネント統合- ポートフォリオ価値計算

        try:- 損益管理

            self.perfect_order_detector = PerfectOrderDetector()- 複雑なエラーハンドリング

            self.ranking_system = HierarchicalRankingSystem()- Excel出力機能

            self.scoring_engine = ComprehensiveScoringEngine()- 決定論モード

            self.switch_manager = IntelligentSwitchManager()- 統計計算エンジン

            self.market_monitor = MarketConditionMonitor()

            ### 開発順序

            self.logger.info("全コンポーネント初期化成功")1. **Phase 1**: 基本クラス作成 + ダミーデータでのテスト

        except Exception as e:2. **Phase 2**: yfinanceによる実データ取得

            self.logger.error(f"コンポーネント初期化失敗: {e}")3. **Phase 3**: パーフェクトオーダー判定ロジック

            raise4. **Phase 4**: ランキング・選択機能

        5. **Phase 5**: 日次実行ループ + 結果出力

        # 銘柄リスト

        self.symbol_universe = ['7203', '9984', '6758', '4063', '8306']### 注意事項・制約

        self.current_position = None- **シンプル第一**: 複雑な機能は後回し

    - **エラーは最小限**: try-except は必要最小限

    def run_daily_selection(self, target_date: datetime) -> Dict[str, Any]:- **ログ出力重視**: 各段階の処理状況を明確にログ

        """日次銘柄選択メイン処理"""- **テスト可能**: 各メソッドが独立してテスト可能

        # Phase 3で実装- **将来拡張性**: マルチ戦略システム統合を意識した設計

        pass

```---



**チェックリスト**:## 既存システムとの関係

- [ ] クラス初期化成功

- [ ] 全コンポーネント初期化確認### 参考にする既存ファイル

- [ ] ログ出力正常- `src/dssms/dssms_backtester.py` - 複雑すぎるが、一部のロジック参考

- [ ] エラーハンドリング動作確認- `src/dssms/perfect_order_detector.py` - パーフェクトオーダー判定参考

- `data_fetcher.py` - データ取得方法参考

### **Phase 3: 段階的コンポーネント統合**

**目標**: 各コンポーネントの順次統合  ### 既存ファイルで除外する機能

**期間**: 2-3日  <details>

<summary>🚫 以下は既存DSSMSの取引実行部分（V3では除外）</summary>

#### **Phase 3-1: データ取得・パーフェクトオーダー**

```python- ~~独自の取引シミュレーション機能~~

def _fetch_market_data(self, target_date: datetime) -> Dict[str, pd.DataFrame]:- ~~ポートフォリオ価値管理~~

    """市場データ取得"""- ~~Entry_Signal/Exit_Signal生成~~

    # yfinanceまたはDSSMSDataManager活用- ~~戦略名ローテーション機能~~

    - ~~統計計算エンジン~~

def _analyze_perfect_order(self, market_data: Dict) -> Dict[str, Any]:- ~~決定論モード~~

    """パーフェクトオーダー分析"""

    return self.perfect_order_detector.analyze_symbols(**理由**: DSS Core はランキング1位選択までに特化

        market_data, target_date

    )</details>

```

---

#### **Phase 3-2: スコアリング・ランキング**

```python## サンプルコード実装例

def _calculate_comprehensive_scores(self, market_data: Dict, 

                                   perfect_order_results: Dict) -> Dict[str, Any]:### メインクラス構造

    """総合スコア計算"""```python

    return self.scoring_engine.calculate_comprehensive_scores(class DSSBacktesterV3:

        market_data, perfect_order_results    """DSS（Dynamic Stock Selector）バックテスター V3"""

    )    

    def __init__(self):

def _rank_symbols(self, scoring_results: Dict, target_date: datetime) -> List[Dict]:        self.logger = setup_logger(__name__)

    """銘柄ランキング"""        # シンプルな5銘柄で開始

    return self.ranking_system.rank_symbols(        self.symbol_universe = ['7203', '9984', '6758', '4063', '8306']

        scoring_results, target_date        

    )    def run_daily_selection(self, target_date: datetime) -> Dict[str, Any]:

```        """

        日次銘柄選択実行（DSS Core メイン処理）

#### **Phase 3-3: 最終選択・切替判定**        

```python        Returns:

def _evaluate_switch_decision(self, ranking_results: List[Dict],             {

                             market_data: Dict) -> Any:                'date': datetime,

    """切替判定"""                'selected_symbol': str,

    return self.switch_manager.evaluate_switch(                'ranking': List[Dict],

        current_symbol=self.current_position,                'execution_time_ms': float

        ranking_results=ranking_results,            }

        market_data=market_data        """

    )        start_time = time.time()

```        

        # 1. データ取得

**チェックリスト**:        market_data = self.fetch_market_data(self.symbol_universe, target_date)

- [ ] データ取得正常動作        

- [ ] パーフェクトオーダー判定結果確認        # 2. パーフェクトオーダースコア計算

- [ ] スコアリング結果妥当性確認        scores = self.calculate_perfect_order_scores(market_data)

- [ ] ランキング順位正常        

- [ ] 切替判定ロジック動作確認        # 3. ランキング

        ranking = self.rank_symbols(scores)

### **Phase 4: 統合テスト・デバッグ**        

**目標**: 全体フロー動作確認・問題修正          # 4. 1位選択

**期間**: 1-2日          selected_symbol = self.select_top_symbol(ranking)

        

**統合テスト内容**:        execution_time = (time.time() - start_time) * 1000

```python        

def test_daily_selection_integration():        return {

    """統合テスト"""            'date': target_date,

    dss = DSSBacktesterV3()            'selected_symbol': selected_symbol,

                'ranking': ranking,

    # 単日テスト            'execution_time_ms': execution_time

    result = dss.run_daily_selection(datetime(2023, 1, 15))        }

    ```

    # 結果検証

    assert 'selected_symbol' in result### パーフェクトオーダー判定例

    assert result['selected_symbol'] in dss.symbol_universe```python

    assert 'ranking' in resultdef calculate_perfect_order_score(self, data: pd.DataFrame) -> float:

    assert len(result['ranking']) == len(dss.symbol_universe)    """

        パーフェクトオーダースコア計算

    print(f"選択銘柄: {result['selected_symbol']}")    

    print(f"ランキング: {result['ranking']}")    Perfect Order: MA5 > MA25 > MA75 かつ 価格 > MA5

```    

    Returns:

**チェックリスト**:        0.0-1.0 のスコア（1.0が完璧なパーフェクトオーダー）

- [ ] 全体フロー正常動作    """

- [ ] エラーハンドリング確認    if len(data) < 75:  # 75日移動平均に必要なデータ不足

- [ ] パフォーマンス測定        return 0.0

- [ ] ログ出力妥当性確認        

- [ ] 設定ファイル影響確認    # 移動平均計算

    ma5 = data['Close'].rolling(5).mean().iloc[-1]

---    ma25 = data['Close'].rolling(25).mean().iloc[-1]

    ma75 = data['Close'].rolling(75).mean().iloc[-1]

## 完成版クラス構造例    current_price = data['Close'].iloc[-1]

    

```python    # パーフェクトオーダー判定

class DSSBacktesterV3:    score = 0.0

    """DSS（Dynamic Stock Selector）バックテスター V3    

        if current_price > ma5:

    既存DSSMSコンポーネントを統合した銘柄選択エンジン        score += 0.25

    """    if ma5 > ma25:

            score += 0.25

    def __init__(self):    if ma25 > ma75:

        self.logger = setup_logger(__name__)        score += 0.25

        self._initialize_components()    

        self.symbol_universe = ['7203', '9984', '6758', '4063', '8306']    # ボーナス: すべての条件を満たす場合

        self.current_position = None    if score == 0.75:

            score = 1.0

    def _initialize_components(self):        

        """コンポーネント初期化"""    return score

        try:```

            self.perfect_order_detector = PerfectOrderDetector()

            self.ranking_system = HierarchicalRankingSystem()### 実行例

            self.scoring_engine = ComprehensiveScoringEngine()```python

            self.switch_manager = IntelligentSwitchManager()# DSS V3 実行例

            self.market_monitor = MarketConditionMonitor()def main():

            self.logger.info("全DSSMSコンポーネント初期化完了")    dss = DSSBacktesterV3()

        except Exception as e:    

            self.logger.error(f"コンポーネント初期化失敗: {e}")    # 単日実行

            raise    result = dss.run_daily_selection(datetime(2023, 1, 15))

        print(f"選択銘柄: {result['selected_symbol']}")

    def run_daily_selection(self, target_date: datetime) -> Dict[str, Any]:    print(f"実行時間: {result['execution_time_ms']:.1f}ms")

        """日次銘柄選択実行"""    

        start_time = time.time()    # 期間実行（将来拡張）

            # results = dss.run_period_selection(start_date, end_date)

        try:```

            # 1. データ取得

            market_data = self._fetch_market_data(target_date)---

            

            # 2. パーフェクトオーダー分析## 開発チェックリスト

            perfect_order_results = self._analyze_perfect_order(market_data, target_date)

            ### Phase 1: 基本実装

            # 3. 総合スコアリング- [ ] クラス構造作成

            scoring_results = self._calculate_comprehensive_scores(- [ ] ログシステム設定

                market_data, perfect_order_results- [ ] 銘柄リスト定義

            )- [ ] ダミーデータでの動作確認

            

            # 4. 階層的ランキング### Phase 2: データ取得

            ranking_results = self._rank_symbols(scoring_results, target_date)- [ ] yfinance連携

            - [ ] データ品質チェック

            # 5. インテリジェント選択- [ ] エラーハンドリング

            selection_result = self._evaluate_switch_decision(

                ranking_results, market_data### Phase 3: パーフェクトオーダー

            )- [ ] 移動平均計算

            - [ ] スコア算出ロジック

            execution_time = (time.time() - start_time) * 1000- [ ] 境界ケース対応

            

            return {### Phase 4: ランキング・選択

                'date': target_date,- [ ] スコア順ソート

                'selected_symbol': selection_result.recommended_symbol,- [ ] 1位選択機能

                'ranking': ranking_results,- [ ] 結果データ構造定義

                'perfect_order_analysis': perfect_order_results,

                'switch_decision': selection_result,### Phase 5: 統合・テスト

                'execution_time_ms': execution_time,- [ ] 全体フロー確認

                'market_condition': self.market_monitor.get_current_condition()- [ ] パフォーマンステスト

            }- [ ] ログ出力検証
            
        except Exception as e:
            self.logger.error(f"日次選択処理エラー [{target_date}]: {e}")
            raise
    
    def _fetch_market_data(self, target_date: datetime) -> Dict[str, pd.DataFrame]:
        """市場データ取得"""
        # 実装詳細
        pass
    
    def _analyze_perfect_order(self, market_data: Dict, target_date: datetime) -> Dict[str, Any]:
        """パーフェクトオーダー分析"""
        return self.perfect_order_detector.analyze_symbols(market_data, target_date)
    
    def _calculate_comprehensive_scores(self, market_data: Dict, 
                                      perfect_order_results: Dict) -> Dict[str, Any]:
        """総合スコア計算"""
        return self.scoring_engine.calculate_comprehensive_scores(
            market_data, perfect_order_results
        )
    
    def _rank_symbols(self, scoring_results: Dict, target_date: datetime) -> List[Dict]:
        """銘柄ランキング"""
        return self.ranking_system.rank_symbols(scoring_results, target_date)
    
    def _evaluate_switch_decision(self, ranking_results: List[Dict], 
                                 market_data: Dict) -> Any:
        """切替判定"""
        return self.switch_manager.evaluate_switch(
            current_symbol=self.current_position,
            ranking_results=ranking_results,
            market_data=market_data
        )

# 実行例
def main():
    dss = DSSBacktesterV3()
    
    # 単日実行
    result = dss.run_daily_selection(datetime(2023, 1, 15))
    print(f"選択銘柄: {result['selected_symbol']}")
    print(f"実行時間: {result['execution_time_ms']:.1f}ms")
    
    # 期間実行（将来拡張）
    # results = dss.run_period_selection(start_date, end_date)

if __name__ == "__main__":
    main()
```

---

## エラー対応・デバッグ指針

### **個別コンポーネントエラー**
```python
def _test_component_individually(self, component_name: str) -> bool:
    """個別コンポーネントテスト"""
    try:
        if component_name == "perfect_order":
            # ダミーデータでテスト
            test_data = self._generate_test_data()
            result = self.perfect_order_detector.analyze_symbols(test_data, datetime.now())
            return result is not None
        elif component_name == "ranking":
            # ダミースコアでテスト
            test_scores = {'7203': 0.8, '9984': 0.6}
            result = self.ranking_system.rank_symbols(test_scores, datetime.now())
            return len(result) > 0
        # 他のコンポーネントも同様
    except Exception as e:
        self.logger.error(f"{component_name} 単体テスト失敗: {e}")
        return False
```

### **設定ファイル問題**
- `config/dssms/ranking_config.json` 読み込みエラー → デフォルト値で継続
- `config/dssms/hierarchical_switch_decision_config.json` 設定ミス → ログ出力して調査

### **データ取得エラー**
- yfinance接続エラー → フォールバック機能実装
- 銘柄データ不足 → 該当銘柄をスキップ

---

## プロジェクト成功指標

### **Phase完了基準**
- **Phase 1**: 全コンポーネントインポート成功率 100%
- **Phase 2**: 統合クラス初期化成功率 100%
- **Phase 3**: 各段階の単体テスト成功率 100%
- **Phase 4**: 統合テスト成功率 95%以上

### **最終成果物**
- [ ] `src/dssms/dssms_backtester_v3.py` 完成
- [ ] 単日選択機能動作確認
- [ ] ログ出力完備
- [ ] エラーハンドリング実装
- [ ] 実行時間1秒以内（5銘柄）
- [ ] 設定ファイル活用確認

**目標**: 実用的なDSS銘柄選択エンジンの完成