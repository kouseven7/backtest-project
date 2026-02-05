"""
シンボルユーティリティ - yfinanceティッカー変換

日本株式のティッカーシンボルをyfinance形式に変換します。
二重サフィックス（.T.T）を防止します。

主な機能:
- 日本株式ティッカーへの .T サフィックス付与
- 既存サフィックスの重複防止
- 空文字列・None のハンドリング
- 前後の空白除去
- 二重サフィックス（.T.T）の自動修正

統合コンポーネント:
- yfinance API: すべてのyfinance呼び出しで使用
- data_fetcher.py: データ取得時のシンボル変換
- data_cache_manager.py: キャッシュ管理時のシンボル変換
- dssms_integrated_main.py: DSSMS統合処理でのシンボル変換

セーフティ機能/注意事項:
- ValueError を発生させる条件: symbol が空文字列または None
- 既に .T で終わる場合は重複を避ける
- .T.T のような二重サフィックスは自動修正
- すべてのyfinance API呼び出しでこの関数を使用すること

Author: Backtest Project Team
Created: 2026-02-05
Last Modified: 2026-02-05
"""


def to_yfinance(symbol: str) -> str:
    """
    ティッカーシンボルをyfinance形式に変換
    
    Args:
        symbol: 銘柄コード（例: "8331" or "8331.T"）
        
    Returns:
        yfinance形式のティッカー（例: "8331.T"）
        
    Examples:
        >>> to_yfinance("8331")
        '8331.T'
        >>> to_yfinance("8331.T")
        '8331.T'
        >>> to_yfinance("8331.T.T")
        '8331.T'  # 二重サフィックスを修正
        
    Note:
        - 既に .T が付いている場合は重複を避ける
        - 空文字列や None の場合は ValueError を発生
        - すべてのyfinance API呼び出しでこの関数を使用すること
        
    Raises:
        ValueError: symbol が空文字列または None の場合
    """
    # 入力チェック
    if not symbol:
        raise ValueError("symbol は空文字列または None にできません")
    
    # 文字列に変換（念のため）
    symbol = str(symbol).strip()
    
    # 既に .T で終わっている場合
    if symbol.endswith('.T'):
        # 二重サフィックス（.T.T）を修正
        while symbol.endswith('.T.T'):
            symbol = symbol[:-2]  # 最後の .T を削除
        return symbol
    
    # .T を付与
    return f"{symbol}.T"


def _test_to_yfinance():
    """to_yfinance() 関数のテスト"""
    # 正常系テスト
    assert to_yfinance("8331") == "8331.T", "通常の銘柄コード"
    assert to_yfinance("8331.T") == "8331.T", "既に .T 付き"
    assert to_yfinance("8331.T.T") == "8331.T", "二重サフィックス修正"
    assert to_yfinance("  8331  ") == "8331.T", "前後の空白除去"
    
    # 異常系テスト
    try:
        to_yfinance("")
        assert False, "空文字列でエラーが発生すべき"
    except ValueError:
        pass
    
    try:
        to_yfinance(None)
        assert False, "None でエラーが発生すべき"
    except ValueError:
        pass
    
    print("[OK] to_yfinance() テスト全て成功")


if __name__ == "__main__":
    _test_to_yfinance()
