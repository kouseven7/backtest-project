#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DSSMS出力問題の完全調査スクリプト
単一データソースからの一貫した出力を確保する調査
"""
import os
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import re

class DSSMSOutputInvestigator:
    def __init__(self):
        self.project_root = Path(".")
        self.results = {}
        
    def investigate_complete_flow(self):
        """完全なデータフロー調査"""
        print("[SEARCH] DSSMS出力システム完全調査開始")
        print("=" * 60)
        
        # 1. 既存出力ファイルの分析
        self._analyze_existing_outputs()
        
        # 2. バックテスターの実際の実行とデータ取得
        self._run_backtester_and_capture_data()
        
        # 3. 各出力システムのデータソース特定
        self._identify_output_data_sources()
        
        # 4. データ不整合の詳細分析
        self._analyze_data_inconsistencies()
        
        # 5. 修正提案の生成
        self._generate_fix_proposals()
        
    def _analyze_existing_outputs(self):
        """既存出力ファイルの分析"""
        print("\n1️⃣ 既存出力ファイル分析")
        
        # 最新の出力ファイルを探索
        output_dirs = [
            'backtest_results/dssms_results',
            'output',
            '.',
            'backtest_results/improved_results'
        ]
        
        found_files = {}
        
        for output_dir in output_dirs:
            if os.path.exists(output_dir):
                try:
                    files = os.listdir(output_dir)
                    
                    # Excel・テキストファイルを検索
                    excel_files = [f for f in files if f.endswith('.xlsx') and ('dssms' in f.lower() or 'backtest' in f.lower())]
                    text_files = [f for f in files if f.endswith('.txt') and ('dssms' in f.lower() or 'report' in f.lower())]
                    
                    if excel_files:
                        for excel_file in excel_files[-3:]:  # 最新3ファイル
                            file_path = os.path.join(output_dir, excel_file)
                            found_files[f'excel_{excel_file}'] = file_path
                    
                    if text_files:
                        for text_file in text_files[-3:]:  # 最新3ファイル
                            file_path = os.path.join(output_dir, text_file)
                            found_files[f'text_{text_file}'] = file_path
                            
                except Exception as e:
                    print(f"   [WARNING] ディレクトリ読み込みエラー {output_dir}: {e}")
        
        print(f"   📄 発見ファイル: {len(found_files)}個")
        for file_type, file_path in found_files.items():
            print(f"     {file_type}: {os.path.basename(file_path)}")
            
        self.results['existing_files'] = found_files
        
        # 最新のExcelファイルを詳細分析
        excel_files = {k: v for k, v in found_files.items() if k.startswith('excel_')}
        if excel_files:
            latest_excel = max(excel_files.items(), key=lambda x: os.path.getmtime(x[1]))
            self._analyze_excel_content(latest_excel[1])
        
    def _analyze_excel_content(self, excel_path):
        """Excelファイルの内容分析"""
        print(f"\n   [CHART] Excel内容分析: {os.path.basename(excel_path)}")
        
        try:
            xl_file = pd.ExcelFile(excel_path)
            print(f"     シート数: {len(xl_file.sheet_names)}")
            print(f"     シート一覧: {xl_file.sheet_names}")
            
            excel_analysis = {}
            
            for sheet_name in xl_file.sheet_names:
                try:
                    df = pd.read_excel(xl_file, sheet_name=sheet_name)
                    excel_analysis[sheet_name] = {
                        'rows': len(df),
                        'columns': len(df.columns),
                        'column_names': list(df.columns),
                        'has_data': not df.empty,
                        'sample_data': df.head(3).to_dict('records') if not df.empty else []
                    }
                    
                    # 特定の問題を検出
                    if 'date' in df.columns or '日付' in df.columns:
                        date_col = 'date' if 'date' in df.columns else '日付'
                        if not df[date_col].isna().all():
                            dates = pd.to_datetime(df[date_col], errors='coerce')
                            if dates.dt.year.max() > 2024:
                                excel_analysis[sheet_name]['date_issue'] = f"2025年データ検出: {dates.dt.year.max()}"
                    
                    print(f"       {sheet_name}: {len(df)}行 x {len(df.columns)}列")
                    
                except Exception as e:
                    print(f"       [ERROR] {sheet_name}読み込みエラー: {e}")
                    excel_analysis[sheet_name] = {'error': str(e)}
            
            self.results['excel_analysis'] = excel_analysis
            
        except Exception as e:
            print(f"     [ERROR] Excel分析エラー: {e}")
            self.results['excel_error'] = str(e)
        
    def _run_backtester_and_capture_data(self):
        """バックテスターを実行してデータを取得"""
        print("\n2️⃣ バックテスター実行とデータ取得")
        
        try:
            # DSSMSバックテスターをインポート
            import sys
            sys.path.append('src')
            
            from src.dssms.dssms_backtester import DSSMSBacktester
            
            print("   [ROCKET] DSSMSバックテスター初期化中...")
            backtester = DSSMSBacktester()
            
            # バックテスター設定の確認
            print("   [LIST] バックテスター設定:")
            print(f"     - クラス: {type(backtester).__name__}")
            print(f"     - 属性数: {len(dir(backtester))}")
            
            # 実行可能メソッドの確認
            available_methods = [method for method in dir(backtester) 
                               if callable(getattr(backtester, method)) 
                               and not method.startswith('_')]
            print(f"     - 利用可能メソッド: {available_methods}")
            
            # バックテスト実行
            if hasattr(backtester, 'run_backtest'):
                print("   🔄 run_backtest()実行中...")
                results = backtester.run_backtest()
                self.results['backtester_raw'] = self._serialize_results(results)
                print(f"   [OK] バックテスト完了: {type(results)}")
                
                # 結果の詳細分析
                if isinstance(results, dict):
                    print("   [CHART] 結果データ構造:")
                    for key, value in results.items():
                        if isinstance(value, pd.DataFrame):
                            print(f"     - {key}: DataFrame ({value.shape})")
                            if not value.empty:
                                print(f"       列: {list(value.columns)}")
                                if hasattr(value.index, 'dtype') and 'datetime' in str(value.index.dtype):
                                    date_range = f"{value.index.min()} ～ {value.index.max()}"
                                    print(f"       期間: {date_range}")
                        elif isinstance(value, (int, float)):
                            print(f"     - {key}: {type(value).__name__} = {value}")
                        elif isinstance(value, dict):
                            print(f"     - {key}: dict ({len(value)}キー)")
                        else:
                            print(f"     - {key}: {type(value).__name__}")
                            
            elif hasattr(backtester, 'simulate_dynamic_selection'):
                print("   🔄 simulate_dynamic_selection()実行中...")
                # デフォルトパラメータで実行
                from datetime import datetime
                results = backtester.simulate_dynamic_selection(
                    start_date=datetime(2023, 1, 1),
                    end_date=datetime(2023, 12, 31)
                )
                self.results['backtester_raw'] = self._serialize_results(results)
                print(f"   [OK] シミュレーション完了: {type(results)}")
                
            else:
                print("   [WARNING] 既知の実行メソッドが見つかりません")
                self.results['backtester_error'] = "実行メソッド未発見"
                
        except ImportError as e:
            print(f"   [ERROR] インポートエラー: {e}")
            self.results['import_error'] = str(e)
        except Exception as e:
            print(f"   [ERROR] バックテスター実行エラー: {e}")
            self.results['backtester_error'] = str(e)
    
    def _serialize_results(self, results):
        """結果データのシリアライズ"""
        if isinstance(results, dict):
            serialized = {}
            for key, value in results.items():
                if isinstance(value, pd.DataFrame):
                    serialized[key] = {
                        'type': 'DataFrame',
                        'shape': value.shape,
                        'columns': list(value.columns),
                        'sample': value.head(3).to_dict('records') if not value.empty else [],
                        'index_type': str(type(value.index)),
                        'dtypes': value.dtypes.to_dict()
                    }
                elif isinstance(value, pd.Series):
                    serialized[key] = {
                        'type': 'Series',
                        'length': len(value),
                        'sample': value.head(3).to_dict()
                    }
                else:
                    serialized[key] = {'type': type(value).__name__, 'value': str(value)}
            return serialized
        return {'type': type(results).__name__, 'value': str(results)}
    
    def _identify_output_data_sources(self):
        """各出力システムのデータソース特定"""
        print("\n3️⃣ 出力システムのデータソース特定")
        
        output_systems = {
            'dssms_backtester': 'src/dssms/dssms_backtester.py',
            'dssms_excel_exporter_v2': 'src/dssms/dssms_excel_exporter_v2.py',
            'simple_excel_exporter': 'output/simple_excel_exporter.py',
            'main_text_reporter': 'output/main_text_reporter.py'
        }
        
        source_analysis = {}
        
        for system_name, file_path in output_systems.items():
            print(f"\n   [SEARCH] {system_name} 分析:")
            
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # データソース特定のキーワード検索
                    analysis = {
                        'file_exists': True,
                        'line_count': len(content.split('\n')),
                        'imports': [],
                        'data_sources': [],
                        'calculation_methods': []
                    }
                    
                    # インポート文の抽出
                    import_lines = [line.strip() for line in content.split('\n') 
                                  if line.strip().startswith('import ') or line.strip().startswith('from ')]
                    analysis['imports'] = import_lines[:5]
                    
                    # データソース関連のキーワード検索
                    data_keywords = ['DataFrame', 'calculate', 'get_', 'fetch_', 'load_', 'portfolio_value', 'trade_history']
                    for keyword in data_keywords:
                        if keyword in content:
                            lines = [line.strip() for line in content.split('\n') 
                                   if keyword in line and not line.strip().startswith('#')]
                            analysis['data_sources'].extend(lines[:2])
                    
                    # 計算メソッドの検索
                    calc_keywords = ['def calculate', 'def get_', 'def process_', 'return']
                    for keyword in calc_keywords:
                        if keyword in content:
                            lines = [line.strip() for line in content.split('\n') 
                                   if keyword in line and 'def ' in line]
                            analysis['calculation_methods'].extend(lines[:3])
                    
                    source_analysis[system_name] = analysis
                    
                    print(f"     ファイル存在: [OK] ({analysis['line_count']}行)")
                    print(f"     データソース候補: {len(analysis['data_sources'])}個")
                    print(f"     計算メソッド: {len(analysis['calculation_methods'])}個")
                    
                except Exception as e:
                    print(f"     [ERROR] ファイル分析エラー: {e}")
                    source_analysis[system_name] = {'error': str(e)}
            else:
                print(f"     [ERROR] ファイル未存在: {file_path}")
                source_analysis[system_name] = {'file_exists': False}
        
        self.results['source_analysis'] = source_analysis
    
    def _analyze_data_inconsistencies(self):
        """データ不整合の詳細分析"""
        print("\n4️⃣ データ不整合詳細分析")
        
        inconsistencies = []
        
        # Excel分析結果からの不整合検出
        if 'excel_analysis' in self.results:
            excel_data = self.results['excel_analysis']
            
            for sheet_name, sheet_data in excel_data.items():
                if isinstance(sheet_data, dict) and 'date_issue' in sheet_data:
                    inconsistencies.append({
                        'type': 'date_inconsistency',
                        'location': f'Excel:{sheet_name}',
                        'issue': sheet_data['date_issue']
                    })
                
                # ゼロ値問題の検出
                if isinstance(sheet_data, dict) and 'sample_data' in sheet_data:
                    sample = sheet_data['sample_data']
                    if sample and len(sample) > 0:
                        first_row = sample[0]
                        zero_fields = [k for k, v in first_row.items() 
                                     if v == 0 or v == '0円' or v == '0.00%']
                        if len(zero_fields) > 3:  # 3つ以上のフィールドが0
                            inconsistencies.append({
                                'type': 'zero_value_issue',
                                'location': f'Excel:{sheet_name}',
                                'issue': f'複数フィールドが0値: {zero_fields}'
                            })
        
        # バックテスター結果との比較
        if 'backtester_raw' in self.results:
            bt_data = self.results['backtester_raw']
            print("   [CHART] バックテスター結果概要:")
            for key, value in bt_data.items():
                if isinstance(value, dict) and value.get('type') == 'DataFrame':
                    print(f"     - {key}: {value['shape']}")
                    if 'sample' in value and value['sample']:
                        sample_keys = list(value['sample'][0].keys()) if value['sample'] else []
                        print(f"       列例: {sample_keys[:5]}")
        
        print(f"   [ALERT] 検出された不整合: {len(inconsistencies)}件")
        for i, issue in enumerate(inconsistencies, 1):
            print(f"     {i}. {issue['type']} @ {issue['location']}: {issue['issue']}")
        
        self.results['inconsistencies'] = inconsistencies
    
    def _generate_fix_proposals(self):
        """修正提案の生成"""
        print("\n5️⃣ 修正提案生成")
        
        proposals = []
        
        # 検出された問題に基づく提案
        if 'inconsistencies' in self.results:
            issues = self.results['inconsistencies']
            
            date_issues = [i for i in issues if i['type'] == 'date_inconsistency']
            zero_issues = [i for i in issues if i['type'] == 'zero_value_issue']
            
            if date_issues:
                proposals.append({
                    'priority': 'High',
                    'title': '日付データ修正',
                    'description': 'バックテスト期間（2023年）と出力データ（2025年）の日付不整合を修正',
                    'action': 'バックテスターの日付管理ロジック修正'
                })
            
            if zero_issues:
                proposals.append({
                    'priority': 'Critical', 
                    'title': 'ゼロ値問題修正',
                    'description': 'サマリーデータが全て0になる問題を修正',
                    'action': 'データフロー統一による計算ロジック修正'
                })
        
        # 根本的解決提案
        proposals.extend([
            {
                'priority': 'Critical',
                'title': '統一出力エンジン構築',
                'description': '単一データソースから全出力形式を生成する統一システム',
                'action': 'DSSMSUnifiedOutputEngine実装'
            },
            {
                'priority': 'High',
                'title': 'データ検証システム',
                'description': '出力前のデータ整合性チェック機能',
                'action': 'ValidationEngine実装'
            },
            {
                'priority': 'Medium',
                'title': '銘柄切り替え機能修正',
                'description': 'DSSMSの動的銘柄選択機能の修正',
                'action': 'SwitchManager見直し'
            }
        ])
        
        print("   推奨修正アプローチ:")
        for i, proposal in enumerate(proposals, 1):
            print(f"   {i}. [{proposal['priority']}] {proposal['title']}")
            print(f"      {proposal['description']}")
            print(f"      → {proposal['action']}")
            print()
        
        self.results['proposals'] = proposals
    
    def generate_report(self):
        """調査レポートの生成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"dssms_output_investigation_{timestamp}.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("DSSMS出力システム調査レポート\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"調査実行時刻: {datetime.now()}\n\n")
            
            f.write("【調査結果サマリー】\n")
            f.write("-" * 30 + "\n")
            for key, value in self.results.items():
                if key == 'inconsistencies':
                    f.write(f"- 検出された不整合: {len(value)}件\n")
                elif key == 'proposals':
                    f.write(f"- 修正提案: {len(value)}件\n")
                elif key == 'existing_files':
                    f.write(f"- 発見ファイル: {len(value)}個\n")
                else:
                    f.write(f"- {key}: 分析済み\n")
            
            f.write(f"\n【詳細データ】\n")
            f.write("-" * 30 + "\n")
            f.write(json.dumps(self.results, indent=2, ensure_ascii=False, default=str))
        
        print(f"\n📄 調査レポート生成: {report_path}")
        return report_path

def main():
    """メイン実行関数"""
    investigator = DSSMSOutputInvestigator()
    investigator.investigate_complete_flow()
    report_path = investigator.generate_report()
    
    print("\n" + "="*60)
    print("[TARGET] 調査完了！次のステップの準備ができました。")
    print("[LIST] 詳細レポート:", report_path)
    print("[ROCKET] 次: 統一出力エンジンの実装")
    print("="*60)

if __name__ == "__main__":
    main()
