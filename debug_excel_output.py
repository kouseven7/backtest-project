#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Excel出力問題診断スクリプト
現在の各エンジンの動作状況を確認
"""
import os
import sys
from pathlib import Path
import warnings
from datetime import datetime

# 警告を抑制
warnings.filterwarnings('ignore')

try:
    import pandas as pd
except ImportError:
    print("[ERROR] pandasがインストールされていません")
    sys.exit(1)

def test_engine_output(engine_name, engine_path):
    """各エンジンの出力をテスト"""
    print(f"\n=== {engine_name} テスト ===")
    
    try:
        # エンジンをインポート
        sys.path.append(str(Path.cwd()))
        
        if engine_name == "original":
            from dssms_unified_output_engine import DSSMSUnifiedOutputEngine
            engine = DSSMSUnifiedOutputEngine()
        elif engine_name == "fixed":
            from dssms_unified_output_engine_fixed import DSSMSUnifiedOutputEngine
            engine = DSSMSUnifiedOutputEngine()
        else:
            print(f"[ERROR] 未知のエンジン: {engine_name}")
            return None
        
        # バックテスト実行
        if engine.run_dssms_backtester_and_capture():
            results = engine.generate_all_outputs()
            
            if results.get('excel'):
                excel_path = results['excel']
                
                # Excelファイルの内容確認
                excel_analysis = analyze_excel_file(excel_path)
                
                print(f"[OK] {engine_name} 成功")
                print(f"   Excel: {excel_path}")
                print(f"   シート数: {excel_analysis['sheet_count']}")
                print(f"   シート名: {excel_analysis['sheet_names']}")
                
                return {
                    'success': True,
                    'excel_path': excel_path,
                    'analysis': excel_analysis,
                    'engine': engine
                }
            else:
                print(f"[ERROR] {engine_name} Excel生成失敗")
                return {'success': False, 'error': 'Excel生成失敗'}
        else:
            print(f"[ERROR] {engine_name} バックテスト失敗")
            return {'success': False, 'error': 'バックテスト失敗'}
            
    except Exception as e:
        print(f"[ERROR] {engine_name} エラー: {e}")
        return {'success': False, 'error': str(e)}

def analyze_excel_file(excel_path):
    """Excelファイルの詳細分析"""
    try:
        # ファイル存在確認
        if not Path(excel_path).exists():
            return {'error': 'ファイルが見つかりません'}
        
        # シート名取得
        excel_file = pd.ExcelFile(excel_path)
        sheet_names = excel_file.sheet_names
        
        analysis = {
            'sheet_count': len(sheet_names),
            'sheet_names': sheet_names,
            'sheet_details': {}
        }
        
        # 各シートの内容確認
        for sheet_name in sheet_names:
            try:
                df = pd.read_excel(excel_path, sheet_name=sheet_name)
                analysis['sheet_details'][sheet_name] = {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'column_names': list(df.columns),
                    'has_data': not df.empty,
                    'sample_data': df.head(3).to_dict('records') if not df.empty else []
                }
            except Exception as e:
                analysis['sheet_details'][sheet_name] = {'error': str(e)}
        
        return analysis
        
    except Exception as e:
        return {'error': str(e)}

def main():
    """診断メイン実行"""
    print("[SEARCH] Excel出力問題診断開始")
    print("=" * 60)
    
    # 利用可能エンジンをテスト
    engines = [
        ("original", "dssms_unified_output_engine.py"),
        ("fixed", "dssms_unified_output_engine_fixed.py")
    ]
    
    results = {}
    
    for engine_name, engine_file in engines:
        if Path(engine_file).exists():
            results[engine_name] = test_engine_output(engine_name, engine_file)
        else:
            print(f"[ERROR] {engine_name}: ファイル {engine_file} が見つかりません")
            results[engine_name] = {'success': False, 'error': 'ファイル未発見'}
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("[CHART] 診断結果サマリー")
    print("=" * 60)
    
    successful_engines = []
    
    for engine_name, result in results.items():
        if result.get('success'):
            successful_engines.append(engine_name)
            print(f"[OK] {engine_name}: 正常動作")
            
            analysis = result.get('analysis', {})
            required_sheets = ['サマリー', 'パフォーマンス指標', '取引履歴', '損益推移', '戦略別統計', '切替分析']
            existing_sheets = analysis.get('sheet_names', [])
            
            missing_sheets = [s for s in required_sheets if s not in existing_sheets]
            if missing_sheets:
                print(f"   [WARNING] 不足シート: {missing_sheets}")
            else:
                print(f"   [OK] 全シート存在")
                
            # データ品質チェック
            sheet_details = analysis.get('sheet_details', {})
            for sheet_name in existing_sheets:
                detail = sheet_details.get(sheet_name, {})
                if detail.get('has_data'):
                    print(f"   [OK] {sheet_name}: {detail['rows']}行のデータ")
                else:
                    print(f"   [ERROR] {sheet_name}: データなし")
        else:
            print(f"[ERROR] {engine_name}: {result.get('error', '不明なエラー')}")
    
    # 推奨アクション
    print(f"\n[TARGET] 推奨アクション:")
    if successful_engines:
        best_engine = successful_engines[0]
        print(f"   1. '{best_engine}' エンジンを基準に修正を進める")
        print(f"   2. 不足シートやデータ問題を個別に修正")
        print(f"   3. From_Score/To_Score問題を最優先で解決")
    else:
        print(f"   1. 全エンジンが失敗 - 基本的な問題を解決")
        print(f"   2. コミットを前回の成功状態に戻すことを検討")
        print(f"   3. 最小限の機能から段階的に修復")
    
    return results

if __name__ == "__main__":
    main()