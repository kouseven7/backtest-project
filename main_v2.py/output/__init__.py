"""
main_v2 出力管理モジュール

Phase 1 対応:
- CSV出力
- JSON出力  
- TXT出力
- Excel出力禁止

再利用予定モジュール (main.py実証済み):
- output.unified_exporter.UnifiedExporter (中優先度)
- output.main_text_reporter.generate_main_text_report (中優先度)

出力要件:
- Entry_Signal/Exit_Signal列必須
- 実際のトレード件数 > 0確認
- profit=0の場合は原因調査
- Unicodeエラー対策(Windows対応)
"""

# TODO: Phase 1実装予定
# 1. UnifiedExporter統合テスト
# 2. CSV+JSON+TXT出力確認
# 3. Excel出力完全禁止
# 4. main_text_reporter統合

class OutputManager:
    """main_v2.py専用出力管理クラス"""
    
    def __init__(self):
        self.phase = "Phase 1"
        self.allowed_formats = ["CSV", "JSON", "TXT"]
        self.forbidden_formats = ["Excel", "XLSX", "XLS"]
        
    def validate_format(self, format_name: str) -> bool:
        """出力形式検証"""
        if format_name.upper() in [f.upper() for f in self.forbidden_formats]:
            raise ValueError(f"禁止された出力形式: {format_name}")
        return format_name.upper() in [f.upper() for f in self.allowed_formats]
        
    def export_results(self, data, format_type: str):
        """結果出力"""
        if not self.validate_format(format_type):
            raise ValueError(f"サポートされていない形式: {format_type}")
        
        # TODO: 実装予定
        # 1. CSV出力
        # 2. JSON出力
        # 3. TXT出力
        print(f"出力形式: {format_type} (Phase 1で実装予定)")