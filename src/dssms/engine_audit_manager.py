"""
DSSMS エンジン監査・整理管理システム
Problem 13: エンジン競合解決

実態調査結果：
- output/配下: 8個のPythonファイル
- src/dssms/配下: 9個のエンジン関連ファイル
- 総エンジン数: 17個（当初想定103個は誤認）
- 85.0点基準エンジン: dssms_unified_output_engine.py
"""

import os
import ast
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EngineQualityMetrics:
    """エンジン品質指標"""
    code_quality_score: float
    functionality_score: float
    maintenance_score: float
    usage_frequency: int
    overall_score: float
    classification: str  # 'adopted', 'archived', 'deprecated'
    file_size_kb: float
    last_modified_days: int
    import_count: int

class EngineAuditManager:
    """エンジン品質評価・整理管理"""
    
    def __init__(self, project_root: str = None):
        if project_root is None:
            project_root = "c:\\Users\\imega\\Documents\\my_backtest_project"
        
        self.project_root = Path(project_root)
        self.engine_registry: Dict[str, EngineQualityMetrics] = {}
        self.quality_standards = {
            'adopted_threshold': 75.0,    # 採用基準点（実態に合わせ調整）
            'archived_threshold': 50.0,   # アーカイブ基準点
            'reference_engine': 'dssms_unified_output_engine.py'
        }
        
        logger.info(f"エンジン監査マネージャ初期化: {self.project_root}")
        
    def audit_all_engines(self) -> Dict[str, EngineQualityMetrics]:
        """全エンジンの品質評価実行"""
        # TODO(tag:phase2, rationale:17個エンジン整理): 品質評価実装
        engines = self._discover_engines()
        
        logger.info(f"発見されたエンジン数: {len(engines)}")
        
        for engine_path in engines:
            try:
                metrics = self._evaluate_engine_quality(engine_path)
                self.engine_registry[str(engine_path)] = metrics
                logger.info(f"評価完了 {engine_path.name}: {metrics.overall_score:.1f}点 ({metrics.classification})")
            except Exception as e:
                logger.error(f"評価失敗 {engine_path}: {e}")
                
        return self.engine_registry
        
    def _discover_engines(self) -> List[Path]:
        """エンジンファイル探索"""
        engine_files = []
        
        # output/配下のPythonファイル
        output_engines = list(self.project_root.glob('output/*.py'))
        output_engines.extend(list(self.project_root.glob('output/**/*.py')))
        
        # src/dssms/配下のエンジン関連ファイル
        dssms_engines = list(self.project_root.glob('src/dssms/*engine*.py'))
        dssms_engines.extend(list(self.project_root.glob('src/dssms/**/*engine*.py')))
        
        # 85.0点基準エンジン（プロジェクトルート）
        unified_engine = self.project_root / 'dssms_unified_output_engine.py'
        if unified_engine.exists():
            engine_files.append(unified_engine)
        
        # 統合・重複削除
        engine_files.extend(output_engines)
        engine_files.extend(dssms_engines)
        
        # __init__.pyを除外
        engine_files = [f for f in engine_files if f.name != '__init__.py']
        
        # 重複削除
        return list(set(engine_files))
        
    def _evaluate_engine_quality(self, engine_path: Path) -> EngineQualityMetrics:
        """エンジン品質評価"""
        # TODO(tag:phase2, rationale:品質基準統一): 詳細評価実装
        
        scores = {
            'code_quality': self._assess_code_quality(engine_path),
            'functionality': self._assess_functionality(engine_path),
            'maintenance': self._assess_maintenance_status(engine_path),
            'usage_frequency': self._assess_usage_frequency(engine_path)
        }
        
        # ファイル情報取得
        file_stats = engine_path.stat()
        file_size_kb = file_stats.st_size / 1024
        last_modified_days = (datetime.now().timestamp() - file_stats.st_mtime) / (24 * 3600)
        
        # 重み付き総合評価
        overall_score = (
            scores['code_quality'] * 0.3 +
            scores['functionality'] * 0.4 +
            scores['maintenance'] * 0.2 +
            min(scores['usage_frequency'] * 10, 100) * 0.1
        )
        
        # 分類決定
        classification = self._determine_classification(overall_score, engine_path)
        
        return EngineQualityMetrics(
            code_quality_score=scores['code_quality'],
            functionality_score=scores['functionality'],
            maintenance_score=scores['maintenance'],
            usage_frequency=scores['usage_frequency'],
            overall_score=overall_score,
            classification=classification,
            file_size_kb=file_size_kb,
            last_modified_days=int(last_modified_days),
            import_count=self._count_imports(engine_path)
        )
        
    def _assess_code_quality(self, engine_path: Path) -> float:
        """コード品質評価"""
        try:
            with open(engine_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # AST解析による基本品質チェック
            try:
                tree = ast.parse(content)
                ast_score = 40.0  # 構文正常
            except SyntaxError:
                return 0.0  # 構文エラー
                
            # コード品質指標
            lines = content.split('\\n')
            non_empty_lines = [line for line in lines if line.strip()]
            
            quality_score = ast_score
            
            # ドキュメント文字列の有無
            if '"""' in content:
                quality_score += 20.0
                
            # クラス定義の有無
            if 'class ' in content:
                quality_score += 15.0
                
            # エラーハンドリングの有無
            if 'try:' in content and 'except' in content:
                quality_score += 10.0
                
            # ロギングの有無
            if 'logging' in content or 'logger' in content:
                quality_score += 10.0
                
            # TODO/FIXMEコメントの有無（メンテナンス性）
            if 'TODO' in content or 'FIXME' in content:
                quality_score += 3.0
                
            # 適切なファイルサイズ
            if 100 <= len(non_empty_lines) <= 500:
                quality_score += 2.0
                
            return min(quality_score, 100.0)
            
        except Exception as e:
            logger.warning(f"コード品質評価失敗 {engine_path}: {e}")
            return 0.0
            
    def _assess_functionality(self, engine_path: Path) -> float:
        """機能性評価"""
        try:
            with open(engine_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            functionality_score = 0.0
            
            # 基本的な出力エンジン機能
            if 'def ' in content:
                functionality_score += 20.0
                
            # Excel出力機能
            if any(term in content.lower() for term in ['excel', 'xlsx', 'openpyxl']):
                functionality_score += 25.0
                
            # データ処理機能
            if 'pandas' in content or 'pd.' in content:
                functionality_score += 20.0
                
            # DSSMS統合機能
            if 'dssms' in content.lower():
                functionality_score += 25.0
                
            # 統計計算機能
            if any(stat in content.lower() for stat in ['mean', 'std', 'sharpe', 'profit_factor', 'drawdown']):
                functionality_score += 10.0
                
            return min(functionality_score, 100.0)
            
        except Exception as e:
            logger.warning(f"機能性評価失敗 {engine_path}: {e}")
            return 0.0
            
    def _assess_maintenance_status(self, engine_path: Path) -> float:
        """メンテナンス状況評価"""
        try:
            stat = engine_path.stat()
            
            # ファイルの更新日時から新しさを評価
            now = datetime.now().timestamp()
            file_age_days = (now - stat.st_mtime) / (24 * 3600)
            
            # 新しいファイルほど高評価
            if file_age_days < 7:
                age_score = 100.0
            elif file_age_days < 30:
                age_score = 80.0
            elif file_age_days < 90:
                age_score = 60.0
            elif file_age_days < 180:
                age_score = 40.0
            else:
                age_score = 20.0
                
            return age_score
            
        except Exception as e:
            logger.warning(f"メンテナンス評価失敗 {engine_path}: {e}")
            return 50.0  # デフォルト値
            
    def _assess_usage_frequency(self, engine_path: Path) -> int:
        """使用頻度評価"""
        try:
            engine_name = engine_path.stem
            usage_count = 0
            
            # プロジェクト全体での参照回数を調査
            for py_file in self.project_root.rglob('*.py'):
                if py_file == engine_path:
                    continue
                    
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # インポート文での参照
                    if f'import {engine_name}' in content:
                        usage_count += 3
                    if f'from {engine_name}' in content:
                        usage_count += 3
                    if f'from output.{engine_name}' in content:
                        usage_count += 3
                    if f'from src.dssms.{engine_name}' in content:
                        usage_count += 3
                    if engine_name in content:
                        usage_count += 1
                        
                except Exception:
                    continue
                    
            return usage_count
            
        except Exception as e:
            logger.warning(f"使用頻度評価失敗 {engine_path}: {e}")
            return 0
            
    def _count_imports(self, engine_path: Path) -> int:
        """インポート数カウント"""
        try:
            with open(engine_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            import_count = content.count('import ') + content.count('from ')
            return import_count
            
        except Exception:
            return 0
            
    def _determine_classification(self, overall_score: float, engine_path: Path) -> str:
        """分類決定"""
        engine_name = engine_path.name
        
        # 85.0点エンジンは必ず採用
        if engine_name == self.quality_standards['reference_engine']:
            return 'adopted'
            
        # DSSMSバックテスターで使用されるエンジンは優先
        if any(term in engine_name.lower() for term in ['dssms', 'unified', 'excel']):
            if overall_score >= 60.0:
                return 'adopted'
                
        # 品質基準による分類
        if overall_score >= self.quality_standards['adopted_threshold']:
            return 'adopted'
        elif overall_score >= self.quality_standards['archived_threshold']:
            return 'archived'
        else:
            return 'deprecated'
            
    def classify_engines(self) -> Dict[str, List[str]]:
        """エンジン分類実行"""
        classification = {
            'adopted': [],
            'archived': [],
            'deprecated': []
        }
        
        for engine_path, metrics in self.engine_registry.items():
            classification[metrics.classification].append(engine_path)
            
        # 85.0点エンジンを必ず adopted に含める
        reference_engine_path = str(self.project_root / self.quality_standards['reference_engine'])
        if reference_engine_path not in classification['adopted']:
            classification['adopted'].append(reference_engine_path)
            
        return classification
        
    def execute_reorganization(self, classification: Dict[str, List[str]], dry_run: bool = True) -> Dict[str, Any]:
        """エンジン再編成実行"""
        # TODO(tag:phase2, rationale:安全性確保): 実際の移動処理実装
        
        reorganization_log = {
            'timestamp': datetime.now().isoformat(),
            'dry_run': dry_run,
            'actions': [],
            'errors': []
        }
        
        # アーカイブディレクトリ作成
        archive_base = self.project_root / 'archive' / 'engines'
        
        if not dry_run:
            archive_base.mkdir(parents=True, exist_ok=True)
            (archive_base / 'historical').mkdir(parents=True, exist_ok=True)
            (archive_base / 'deprecated').mkdir(parents=True, exist_ok=True)
            
        # アーカイブ処理
        for engine_path in classification['archived']:
            try:
                dest_path = archive_base / 'historical' / Path(engine_path).name
                
                if not dry_run:
                    shutil.copy2(engine_path, dest_path)
                    
                action = {
                    'type': 'archive',
                    'source': engine_path,
                    'destination': str(dest_path),
                    'status': 'success',
                    'dry_run': dry_run
                }
                reorganization_log['actions'].append(action)
                
            except Exception as e:
                error = {
                    'type': 'archive_error',
                    'file': engine_path,
                    'error': str(e)
                }
                reorganization_log['errors'].append(error)
                
        # 削除処理（deprecated）
        for engine_path in classification['deprecated']:
            try:
                dest_path = archive_base / 'deprecated' / Path(engine_path).name
                
                if not dry_run:
                    # バックアップ作成
                    shutil.copy2(engine_path, dest_path)
                    # 元ファイル削除
                    os.remove(engine_path)
                    
                action = {
                    'type': 'remove',
                    'source': engine_path,
                    'backup': str(dest_path),
                    'status': 'success',
                    'dry_run': dry_run
                }
                reorganization_log['actions'].append(action)
                
            except Exception as e:
                error = {
                    'type': 'remove_error',
                    'file': engine_path,
                    'error': str(e)
                }
                reorganization_log['errors'].append(error)
                
        return reorganization_log
        
    def generate_audit_report(self) -> str:
        """監査レポート生成"""
        classification = self.classify_engines()
        
        report = f"""
# エンジン監査レポート (Problem 13)
Generated: {datetime.now().isoformat()}

## 統計サマリー
- **総エンジン数**: {len(self.engine_registry)}
- **採用エンジン**: {len(classification['adopted'])}
- **アーカイブ対象**: {len(classification['archived'])}
- **削除対象**: {len(classification['deprecated'])}
- **整理率**: {((len(classification['archived']) + len(classification['deprecated'])) / len(self.engine_registry) * 100):.1f}%

## 採用エンジン一覧 (85.0点基準)
"""
        for engine_path in classification['adopted']:
            metrics = self.engine_registry.get(engine_path)
            if metrics:
                path_obj = Path(engine_path)
                report += f"- {path_obj.name} (品質: {metrics.overall_score:.1f}点, サイズ: {metrics.file_size_kb:.1f}KB)\\n"
                
        report += """
## アーカイブ対象エンジン
"""
        for engine_path in classification['archived']:
            metrics = self.engine_registry.get(engine_path)
            if metrics:
                path_obj = Path(engine_path)
                report += f"- {path_obj.name} (品質: {metrics.overall_score:.1f}点)\\n"
                
        report += """
## 削除対象エンジン
"""
        for engine_path in classification['deprecated']:
            metrics = self.engine_registry.get(engine_path)
            if metrics:
                path_obj = Path(engine_path)
                report += f"- {path_obj.name} (品質: {metrics.overall_score:.1f}点)\\n"
                
        report += """
## 品質統計
"""
        scores = [m.overall_score for m in self.engine_registry.values()]
        if scores:
            report += f"- **平均品質**: {sum(scores)/len(scores):.1f}点\\n"
            report += f"- **最高品質**: {max(scores):.1f}点\\n"
            report += f"- **最低品質**: {min(scores):.1f}点\\n"
            report += f"- **85.0点基準エンジン**: {self.quality_standards['reference_engine']}\\n"
            
        return report
        
    def save_report_to_file(self, report: str, filename: str = None) -> str:
        """レポートをファイルに保存"""
        if filename is None:
            filename = f"engine_audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            
        report_path = self.project_root / 'docs' / filename
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
            
        logger.info(f"監査レポート保存: {report_path}")
        return str(report_path)

def main():
    """メイン実行関数"""
    logger.info("=== Problem 13: エンジン競合解決 開始 ===")
    
    # エンジン監査マネージャ初期化
    audit_manager = EngineAuditManager()
    
    # 全エンジン品質評価
    engine_metrics = audit_manager.audit_all_engines()
    logger.info(f"品質評価完了: {len(engine_metrics)}個のエンジン")
    
    # エンジン分類
    classification = audit_manager.classify_engines()
    
    # レポート生成
    report = audit_manager.generate_audit_report()
    print(report)
    
    # レポート保存
    report_path = audit_manager.save_report_to_file(report)
    
    logger.info("=== Problem 13: エンジン監査完了 ===")
    
    return {
        'engine_count': len(engine_metrics),
        'classification': classification,
        'report_path': report_path
    }

if __name__ == "__main__":
    main()