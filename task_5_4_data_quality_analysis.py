#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Task 5.4: データ品質調査
データ取得・前処理品質分析、possibly delisted警告、データ欠損、品質統計の詳細調査
"""

import os
import sys
import json
import re
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import warnings

class CustomJSONEncoder(json.JSONEncoder):
    """NumPy型に対応するカスタムJSONエンコーダー"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        return super().default(obj)

def analyze_data_quality():
    """データ品質分析実行"""
    print("[SEARCH] Task 5.4: データ品質調査開始")
    
    analysis_results = {
        "analysis_timestamp": datetime.now().isoformat(),
        "task": "5.4", 
        "purpose": "データ取得・前処理品質分析",
        "data_sources_analyzed": [],
        "delisting_warnings": [],
        "data_completeness": {},
        "quality_statistics": {},
        "error_patterns": [],
        "data_corruption_issues": [],
        "market_data_gaps": [],
        "data_preprocessing_issues": [],
        "impact_assessment": {},
        "summary_findings": {}
    }
    
    # 1. データソース検出と分析
    print("\n[CHART] Step 1: データソース検出")
    data_sources = detect_data_sources()
    analysis_results["data_sources_analyzed"] = data_sources
    
    # 2. Delisting警告分析
    print("\n[WARNING] Step 2: Delisting警告分析")
    delisting_analysis = analyze_delisting_warnings()
    analysis_results["delisting_warnings"] = delisting_analysis
    
    # 3. データ完全性分析
    print("\n[UP] Step 3: データ完全性分析")
    completeness_analysis = analyze_data_completeness()
    analysis_results["data_completeness"] = completeness_analysis
    
    # 4. データ品質統計
    print("\n[CHART] Step 4: データ品質統計")
    quality_stats = analyze_quality_statistics()
    analysis_results["quality_statistics"] = quality_stats
    
    # 5. エラーパターン分析
    print("\n[SEARCH] Step 5: エラーパターン分析")
    error_patterns = analyze_error_patterns()
    analysis_results["error_patterns"] = error_patterns
    
    # 6. データ破損問題検出
    print("\n💥 Step 6: データ破損問題検出")
    corruption_issues = analyze_data_corruption()
    analysis_results["data_corruption_issues"] = corruption_issues
    
    # 7. 市場データギャップ分析
    print("\n📅 Step 7: 市場データギャップ分析")
    gap_analysis = analyze_market_data_gaps()
    analysis_results["market_data_gaps"] = gap_analysis
    
    # 8. 前処理問題分析
    print("\n⚙️ Step 8: 前処理問題分析")
    preprocessing_analysis = analyze_preprocessing_issues()
    analysis_results["data_preprocessing_issues"] = preprocessing_analysis
    
    # 9. 影響評価
    print("\n[CHART] Step 9: 影響評価")
    impact_assessment = assess_impact_on_switching(analysis_results)
    analysis_results["impact_assessment"] = impact_assessment
    
    # 10. 総合分析
    print("\n[LIST] Step 10: 総合分析")
    summary = generate_summary_analysis(analysis_results)
    analysis_results["summary_findings"] = summary
    
    # 結果保存
    output_file = "task_5_4_data_quality_analysis_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=2, cls=CustomJSONEncoder)
    
    print(f"\n[OK] Task 5.4 完了: {output_file}")
    print(f"[CHART] データソース数: {len(data_sources)}")
    print(f"[WARNING] Delisting警告: {len(delisting_analysis)}")
    print(f"💥 破損問題: {len(corruption_issues)}")
    print(f"📅 データギャップ: {len(gap_analysis)}")
    
    return analysis_results

def detect_data_sources() -> List[Dict[str, Any]]:
    """データソースの検出と分析"""
    data_sources = []
    
    # 1. CSVファイル検出
    csv_files = find_files_by_pattern("**/*.csv")
    for csv_file in csv_files:
        source_info = analyze_csv_data_source(csv_file)
        if source_info:
            data_sources.append(source_info)
    
    # 2. データフェッチャー分析
    fetcher_files = find_files_by_pattern("**/data_fetcher*.py") + \
                   find_files_by_pattern("**/fetcher*.py")
    for fetcher_file in fetcher_files:
        source_info = analyze_fetcher_source(fetcher_file)
        if source_info:
            data_sources.append(source_info)
    
    # 3. データプロセッサー分析
    processor_files = find_files_by_pattern("**/data_processor*.py") + \
                     find_files_by_pattern("**/processor*.py")
    for processor_file in processor_files:
        source_info = analyze_processor_source(processor_file)
        if source_info:
            data_sources.append(source_info)
    
    return data_sources

def find_files_by_pattern(pattern: str) -> List[str]:
    """パターンマッチングでファイル検索"""
    from glob import glob
    return glob(pattern, recursive=True)

def analyze_csv_data_source(file_path: str) -> Optional[Dict[str, Any]]:
    """CSVデータソースの分析"""
    try:
        # ファイル情報
        file_size = os.path.getsize(file_path)
        
        # CSVサンプル読み込み
        sample_df = pd.read_csv(file_path, nrows=100)
        
        return {
            "source_type": "csv_file",
            "file_path": file_path,
            "file_size_bytes": file_size,
            "columns": list(sample_df.columns),
            "sample_row_count": len(sample_df),
            "data_types": {col: str(dtype) for col, dtype in sample_df.dtypes.to_dict().items()},
            "null_counts": {col: int(count) for col, count in sample_df.isnull().sum().to_dict().items()},
            "memory_usage": int(sample_df.memory_usage(deep=True).sum())
        }
    except Exception as e:
        print(f"[WARNING] CSV分析エラー {file_path}: {e}")
        return None

def analyze_fetcher_source(file_path: str) -> Optional[Dict[str, Any]]:
    """データフェッチャーソースの分析"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # API使用検出
        api_patterns = [
            r'yfinance',
            r'yahoo.*finance',
            r'alpha.*vantage',
            r'quandl',
            r'pandas_datareader',
            r'requests\.get'
        ]
        
        apis_used = []
        for pattern in api_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                apis_used.append(pattern)
        
        # エラーハンドリング検出
        error_handling = {
            "try_except_blocks": len(re.findall(r'try\s*:', content)),
            "error_logging": len(re.findall(r'\.log|logging', content)),
            "exception_handling": len(re.findall(r'except.*:', content))
        }
        
        return {
            "source_type": "fetcher",
            "file_path": file_path,
            "file_size_bytes": os.path.getsize(file_path),
            "apis_used": apis_used,
            "error_handling": error_handling,
            "has_caching": "cache" in content.lower(),
            "has_retry_logic": "retry" in content.lower()
        }
    except Exception as e:
        print(f"[WARNING] Fetcher分析エラー {file_path}: {e}")
        return None

def analyze_processor_source(file_path: str) -> Optional[Dict[str, Any]]:
    """データプロセッサーソースの分析"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 前処理操作検出
        processing_operations = {
            "data_cleaning": len(re.findall(r'dropna|fillna|drop_duplicates', content)),
            "normalization": len(re.findall(r'normalize|standardize|scale', content)),
            "feature_engineering": len(re.findall(r'rolling|ewm|pct_change|diff', content)),
            "filtering": len(re.findall(r'filter|query|loc\[|iloc\[', content)),
            "aggregation": len(re.findall(r'groupby|agg|sum\(|mean\(', content))
        }
        
        return {
            "source_type": "processor",
            "file_path": file_path,
            "file_size_bytes": os.path.getsize(file_path),
            "processing_operations": processing_operations,
            "total_operations": sum(processing_operations.values())
        }
    except Exception as e:
        print(f"[WARNING] Processor分析エラー {file_path}: {e}")
        return None

def analyze_delisting_warnings() -> List[Dict[str, Any]]:
    """Delisting警告の分析"""
    delisting_warnings = []
    
    # ログファイルからdelisting警告を検索
    log_files = find_files_by_pattern("**/*.log") + \
               find_files_by_pattern("**/logs/**/*.txt")
    
    delisting_patterns = [
        r'possibly delisted',
        r'delisted.*symbol',
        r'ticker.*not.*found',
        r'no.*data.*found.*for.*symbol',
        r'invalid.*ticker',
        r'symbol.*discontinued'
    ]
    
    for log_file in log_files[:10]:  # 最初の10ファイルを分析
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            for pattern in delisting_patterns:
                matches = re.findall(pattern + r'.*', content, re.IGNORECASE)
                for match in matches:
                    delisting_warnings.append({
                        "log_file": log_file,
                        "pattern": pattern,
                        "warning_text": match[:200],  # 最初の200文字
                        "severity": determine_delisting_severity(match)
                    })
        except Exception as e:
            print(f"[WARNING] ログ分析エラー {log_file}: {e}")
    
    # コードファイルからdelisting関連処理を検索
    code_files = find_files_by_pattern("**/*.py")
    for code_file in code_files[:20]:  # 最初の20ファイルを分析
        try:
            with open(code_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if any(re.search(pattern, content, re.IGNORECASE) for pattern in delisting_patterns):
                delisting_warnings.append({
                    "source_file": code_file,
                    "type": "code_handling",
                    "has_delisting_handling": True,
                    "handling_quality": assess_delisting_handling_quality(content)
                })
        except Exception as e:
            continue
    
    return delisting_warnings

def determine_delisting_severity(warning_text: str) -> str:
    """Delisting警告の深刻度判定"""
    if "error" in warning_text.lower():
        return "high"
    elif "possibly" in warning_text.lower():
        return "medium"
    else:
        return "low"

def assess_delisting_handling_quality(content: str) -> str:
    """Delisting処理品質の評価"""
    quality_indicators = 0
    
    if "try:" in content and "except" in content:
        quality_indicators += 1
    if "logging" in content or ".log" in content:
        quality_indicators += 1
    if "fallback" in content.lower() or "alternative" in content.lower():
        quality_indicators += 1
    if "retry" in content.lower():
        quality_indicators += 1
    
    if quality_indicators >= 3:
        return "good"
    elif quality_indicators >= 2:
        return "medium"
    else:
        return "poor"

def analyze_data_completeness() -> Dict[str, Any]:
    """データ完全性の分析"""
    completeness = {
        "csv_files_analysis": [],
        "overall_completeness_score": 0.0,
        "missing_data_patterns": [],
        "temporal_gaps": []
    }
    
    # CSVファイルの完全性分析
    csv_files = find_files_by_pattern("**/*.csv")
    
    total_completeness = 0.0
    analyzed_files = 0
    
    for csv_file in csv_files[:10]:  # 最初の10ファイルを分析
        try:
            df = pd.read_csv(csv_file, nrows=1000)  # 最初の1000行を分析
            
            # 欠損率計算
            missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
            completeness_score = 1.0 - missing_ratio
            
            # 時系列データの場合、日付ギャップを検出
            date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            temporal_gaps = []
            
            if date_columns:
                for date_col in date_columns:
                    try:
                        df[date_col] = pd.to_datetime(df[date_col])
                        date_gaps = find_date_gaps(df[date_col])
                        temporal_gaps.extend(date_gaps)
                    except:
                        pass
            
            file_analysis = {
                "file_path": csv_file,
                "completeness_score": completeness_score,
                "missing_ratio": missing_ratio,
                "total_cells": df.shape[0] * df.shape[1],
                "missing_cells": df.isnull().sum().sum(),
                "temporal_gaps": temporal_gaps
            }
            
            completeness["csv_files_analysis"].append(file_analysis)
            total_completeness += completeness_score
            analyzed_files += 1
            
        except Exception as e:
            print(f"[WARNING] 完全性分析エラー {csv_file}: {e}")
    
    if analyzed_files > 0:
        completeness["overall_completeness_score"] = total_completeness / analyzed_files
    
    return completeness

def find_date_gaps(date_series: pd.Series) -> List[Dict[str, Any]]:
    """日付ギャップの検出"""
    gaps = []
    
    try:
        # 日付をソート
        sorted_dates = date_series.dropna().sort_values()
        
        if len(sorted_dates) < 2:
            return gaps
        
        # 1日以上のギャップを検出
        for i in range(1, len(sorted_dates)):
            gap_days = (sorted_dates.iloc[i] - sorted_dates.iloc[i-1]).days
            if gap_days > 1:  # 1日以上のギャップ
                gaps.append({
                    "gap_start": sorted_dates.iloc[i-1].isoformat(),
                    "gap_end": sorted_dates.iloc[i].isoformat(),
                    "gap_days": gap_days
                })
    except Exception as e:
        print(f"[WARNING] 日付ギャップ検出エラー: {e}")
    
    return gaps

def analyze_quality_statistics() -> Dict[str, Any]:
    """データ品質統計の分析"""
    quality_stats = {
        "data_freshness": {},
        "data_accuracy_indicators": {},
        "data_consistency": {},
        "outlier_detection": {}
    }
    
    # データ新鮮度分析
    csv_files = find_files_by_pattern("**/*.csv")
    file_ages = []
    
    for csv_file in csv_files[:5]:
        try:
            file_mtime = os.path.getmtime(csv_file)
            file_age_days = (datetime.now().timestamp() - file_mtime) / (24 * 3600)
            file_ages.append(file_age_days)
        except:
            pass
    
    if file_ages:
        quality_stats["data_freshness"] = {
            "average_age_days": np.mean(file_ages),
            "oldest_file_days": max(file_ages),
            "newest_file_days": min(file_ages),
            "freshness_score": calculate_freshness_score(file_ages)
        }
    
    # データ精度指標
    for csv_file in csv_files[:3]:
        try:
            df = pd.read_csv(csv_file, nrows=500)
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_columns) > 0:
                # 外れ値検出
                outliers = detect_outliers(df[numeric_columns])
                quality_stats["outlier_detection"][csv_file] = outliers
        except:
            pass
    
    return quality_stats

def calculate_freshness_score(file_ages: List[float]) -> float:
    """データ新鮮度スコアの計算"""
    if not file_ages:
        return 0.0
    
    avg_age = np.mean(file_ages)
    
    # 1日以内: 100点、30日以内: 70点、それ以上: 30点以下
    if avg_age <= 1:
        return 100.0
    elif avg_age <= 30:
        return 100.0 - (avg_age - 1) * 30 / 29
    else:
        return max(0, 30.0 - (avg_age - 30) * 30 / 365)

def detect_outliers(df: pd.DataFrame) -> Dict[str, Any]:
    """外れ値検出"""
    outliers = {}
    
    for column in df.columns:
        try:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_count = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
            outlier_ratio = outlier_count / len(df)
            
            outliers[column] = {
                "outlier_count": int(outlier_count),
                "outlier_ratio": float(outlier_ratio),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound)
            }
        except:
            pass
    
    return outliers

def analyze_error_patterns() -> List[Dict[str, Any]]:
    """エラーパターンの分析"""
    error_patterns = []
    
    # ログファイルからエラーパターンを抽出
    log_files = find_files_by_pattern("**/*.log")
    
    error_types = [
        "ConnectionError",
        "TimeoutError", 
        "KeyError",
        "ValueError",
        "HTTPError",
        "SSLError",
        "JSONDecodeError"
    ]
    
    for log_file in log_files[:5]:
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            for error_type in error_types:
                error_count = len(re.findall(error_type, content, re.IGNORECASE))
                if error_count > 0:
                    error_patterns.append({
                        "log_file": log_file,
                        "error_type": error_type,
                        "occurrence_count": error_count,
                        "severity": classify_error_severity(error_type)
                    })
        except Exception as e:
            continue
    
    return error_patterns

def classify_error_severity(error_type: str) -> str:
    """エラーの深刻度分類"""
    high_severity = ["ConnectionError", "TimeoutError", "SSLError"]
    medium_severity = ["HTTPError", "JSONDecodeError"]
    
    if error_type in high_severity:
        return "high"
    elif error_type in medium_severity:
        return "medium"
    else:
        return "low"

def analyze_data_corruption() -> List[Dict[str, Any]]:
    """データ破損問題の検出"""
    corruption_issues = []
    
    csv_files = find_files_by_pattern("**/*.csv")
    
    for csv_file in csv_files[:5]:
        try:
            # ファイル読み込みテスト
            df = pd.read_csv(csv_file, nrows=100)
            
            issues = []
            
            # 1. 不正な文字エンコーディング
            try:
                with open(csv_file, 'r', encoding='utf-8') as f:
                    content = f.read(1000)
            except UnicodeDecodeError:
                issues.append("encoding_error")
            
            # 2. 不正なCSV構造
            if df.shape[1] <= 1:  # 列が1つ以下
                issues.append("invalid_csv_structure")
            
            # 3. 異常な値の検出
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if np.isinf(df[col]).any():
                    issues.append(f"infinite_values_in_{col}")
                if (df[col] == 0).all() and len(df) > 10:
                    issues.append(f"all_zero_values_in_{col}")
            
            if issues:
                corruption_issues.append({
                    "file_path": csv_file,
                    "corruption_types": issues,
                    "severity": "high" if len(issues) >= 3 else "medium"
                })
                
        except Exception as e:
            corruption_issues.append({
                "file_path": csv_file,
                "corruption_types": ["read_error"],
                "error_message": str(e),
                "severity": "high"
            })
    
    return corruption_issues

def analyze_market_data_gaps() -> List[Dict[str, Any]]:
    """市場データギャップの分析"""
    gaps = []
    
    # 実際の市場データファイルを分析
    potential_market_files = [f for f in find_files_by_pattern("**/*.csv") 
                             if any(keyword in f.lower() for keyword in 
                                   ['price', 'stock', 'market', 'ticker', 'quote'])]
    
    for file_path in potential_market_files[:3]:
        try:
            df = pd.read_csv(file_path, nrows=500)
            
            # 日付列を探す
            date_columns = [col for col in df.columns 
                          if any(keyword in col.lower() for keyword in 
                                ['date', 'time', 'timestamp'])]
            
            if date_columns:
                date_col = date_columns[0]
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                
                # 営業日ベースでのギャップ検出
                business_days_gaps = detect_business_day_gaps(df[date_col])
                
                if business_days_gaps:
                    gaps.append({
                        "file_path": file_path,
                        "date_column": date_col,
                        "business_day_gaps": business_days_gaps,
                        "gap_impact": assess_gap_impact(business_days_gaps)
                    })
        except Exception as e:
            continue
    
    return gaps

def detect_business_day_gaps(date_series: pd.Series) -> List[Dict[str, Any]]:
    """営業日ギャップの検出"""
    gaps = []
    
    try:
        # NaN値を除去してソート
        clean_dates = date_series.dropna().sort_values()
        
        if len(clean_dates) < 2:
            return gaps
        
        # 営業日基準での期待日数とのギャップ
        for i in range(1, min(len(clean_dates), 50)):  # 最初の50データポイントを分析
            current_date = clean_dates.iloc[i]
            previous_date = clean_dates.iloc[i-1]
            
            # 営業日数を計算
            business_days = pd.bdate_range(previous_date, current_date).shape[0] - 1
            actual_gap = (current_date - previous_date).days
            
            if business_days > 5:  # 5営業日以上のギャップ
                gaps.append({
                    "gap_start": previous_date.isoformat(),
                    "gap_end": current_date.isoformat(),
                    "business_days_missing": business_days,
                    "calendar_days": actual_gap
                })
    except Exception as e:
        print(f"[WARNING] 営業日ギャップ検出エラー: {e}")
    
    return gaps

def assess_gap_impact(gaps: List[Dict[str, Any]]) -> str:
    """ギャップ影響の評価"""
    if not gaps:
        return "none"
    
    total_missing_days = sum(gap["business_days_missing"] for gap in gaps)
    
    if total_missing_days > 50:
        return "critical"
    elif total_missing_days > 20:
        return "high"
    elif total_missing_days > 5:
        return "medium"
    else:
        return "low"

def analyze_preprocessing_issues() -> List[Dict[str, Any]]:
    """前処理問題の分析"""
    issues = []
    
    # data_processor.pyやdata_fetcher.pyなどの前処理ファイルを分析
    processor_files = find_files_by_pattern("**/data_processor*.py") + \
                     find_files_by_pattern("**/data_fetcher*.py")
    
    for file_path in processor_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            detected_issues = []
            
            # 1. エラーハンドリングの不備
            try_count = len(re.findall(r'try\s*:', content))
            except_count = len(re.findall(r'except.*:', content))
            
            if try_count > except_count:
                detected_issues.append("incomplete_error_handling")
            
            # 2. データ検証の不備
            validation_patterns = [
                r'assert\s+',
                r'\.shape\[',
                r'len\(',
                r'is.*None',
                r'\.empty'
            ]
            
            validation_count = sum(len(re.findall(pattern, content)) 
                                 for pattern in validation_patterns)
            
            if validation_count < 3:
                detected_issues.append("insufficient_data_validation")
            
            # 3. パフォーマンス問題
            if '.copy()' not in content and 'df' in content:
                detected_issues.append("potential_memory_issue")
            
            if detected_issues:
                issues.append({
                    "file_path": file_path,
                    "preprocessing_issues": detected_issues,
                    "severity": assess_preprocessing_severity(detected_issues)
                })
                
        except Exception as e:
            continue
    
    return issues

def assess_preprocessing_severity(issues: List[str]) -> str:
    """前処理問題の深刻度評価"""
    if len(issues) >= 3:
        return "high"
    elif len(issues) >= 2:
        return "medium"
    else:
        return "low"

def assess_impact_on_switching(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """データ品質が切替に与える影響の評価"""
    
    # 各要因の影響度計算
    delisting_impact = calculate_delisting_impact(analysis_results["delisting_warnings"])
    completeness_impact = calculate_completeness_impact(analysis_results["data_completeness"])
    quality_impact = calculate_quality_impact(analysis_results["quality_statistics"])
    corruption_impact = calculate_corruption_impact(analysis_results["data_corruption_issues"])
    gap_impact = calculate_gap_impact_on_switching(analysis_results["market_data_gaps"])
    
    total_impact = delisting_impact + completeness_impact + quality_impact + \
                  corruption_impact + gap_impact
    
    impact_level = "critical" if total_impact >= 40 else \
                  "high" if total_impact >= 25 else \
                  "medium" if total_impact >= 15 else "low"
    
    return {
        "delisting_warnings_impact_percent": delisting_impact,
        "data_completeness_impact_percent": completeness_impact,
        "data_quality_impact_percent": quality_impact,
        "data_corruption_impact_percent": corruption_impact,
        "market_gaps_impact_percent": gap_impact,
        "total_switching_impact_percent": total_impact,
        "impact_level": impact_level,
        "primary_data_issues": identify_primary_data_issues(analysis_results),
        "mitigation_potential": {
            "data_cleaning": min(20, total_impact * 0.5),
            "alternative_sources": min(15, delisting_impact * 0.8),
            "gap_filling": min(10, gap_impact * 0.7)
        }
    }

def calculate_delisting_impact(warnings: List[Dict[str, Any]]) -> float:
    """Delisting警告の影響計算"""
    if not warnings:
        return 0.0
    
    high_severity_count = sum(1 for w in warnings if w.get("severity") == "high")
    medium_severity_count = sum(1 for w in warnings if w.get("severity") == "medium")
    
    impact = high_severity_count * 5 + medium_severity_count * 2
    return min(15, impact)  # 最大15%

def calculate_completeness_impact(completeness: Dict[str, Any]) -> float:
    """データ完全性の影響計算"""
    score = completeness.get("overall_completeness_score", 1.0)
    
    # 完全性スコアが低いほど影響大
    impact = (1.0 - score) * 20
    return min(20, impact)  # 最大20%

def calculate_quality_impact(quality_stats: Dict[str, Any]) -> float:
    """データ品質の影響計算"""
    freshness = quality_stats.get("data_freshness", {})
    freshness_score = freshness.get("freshness_score", 100)
    
    # 新鮮度スコアが低いほど影響大
    impact = (100 - freshness_score) * 0.1
    return min(10, impact)  # 最大10%

def calculate_corruption_impact(corruption_issues: List[Dict[str, Any]]) -> float:
    """データ破損の影響計算"""
    if not corruption_issues:
        return 0.0
    
    high_severity_count = sum(1 for issue in corruption_issues 
                             if issue.get("severity") == "high")
    
    impact = high_severity_count * 8
    return min(25, impact)  # 最大25%

def calculate_gap_impact_on_switching(gaps: List[Dict[str, Any]]) -> float:
    """データギャップの切替への影響計算"""
    if not gaps:
        return 0.0
    
    total_impact = 0
    for gap in gaps:
        gap_severity = gap.get("gap_impact", "low")
        if gap_severity == "critical":
            total_impact += 10
        elif gap_severity == "high":
            total_impact += 6
        elif gap_severity == "medium":
            total_impact += 3
    
    return min(15, total_impact)  # 最大15%

def identify_primary_data_issues(analysis_results: Dict[str, Any]) -> List[str]:
    """主要なデータ問題の特定"""
    issues = []
    
    # Delisting警告
    delisting_count = len(analysis_results["delisting_warnings"])
    if delisting_count > 5:
        issues.append(f"大量のDelisting警告: {delisting_count}件")
    
    # データ完全性
    completeness_score = analysis_results["data_completeness"].get("overall_completeness_score", 1.0)
    if completeness_score < 0.8:
        issues.append(f"データ完全性低下: {completeness_score*100:.1f}%")
    
    # データ破損
    corruption_count = len(analysis_results["data_corruption_issues"])
    if corruption_count > 0:
        issues.append(f"データ破損: {corruption_count}ファイル")
    
    # 市場データギャップ
    gap_count = len(analysis_results["market_data_gaps"])
    if gap_count > 0:
        issues.append(f"市場データギャップ: {gap_count}ファイル")
    
    return issues

def generate_summary_analysis(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """総合分析の生成"""
    
    data_sources = analysis_results["data_sources_analyzed"]
    delisting_warnings = analysis_results["delisting_warnings"]
    completeness = analysis_results["data_completeness"]
    quality_stats = analysis_results["quality_statistics"]
    corruption_issues = analysis_results["data_corruption_issues"]
    impact_assessment = analysis_results["impact_assessment"]
    
    # 全体品質スコア計算
    overall_quality_score = calculate_overall_quality_score(analysis_results)
    
    # 改善推奨事項
    recommendations = generate_data_quality_recommendations(analysis_results)
    
    return {
        "total_data_sources": len(data_sources),
        "data_source_types": count_source_types(data_sources),
        "delisting_warnings_count": len(delisting_warnings),
        "data_corruption_files": len(corruption_issues),
        "overall_quality_score": overall_quality_score,
        "data_completeness_score": completeness.get("overall_completeness_score", 0.0),
        "switching_impact_level": impact_assessment.get("impact_level", "unknown"),
        "total_switching_impact_percent": impact_assessment.get("total_switching_impact_percent", 0),
        "primary_concerns": identify_primary_data_issues(analysis_results),
        "improvement_recommendations": recommendations,
        "data_health_status": assess_data_health_status(overall_quality_score)
    }

def calculate_overall_quality_score(analysis_results: Dict[str, Any]) -> float:
    """全体品質スコアの計算"""
    
    # 基本スコア
    base_score = 100.0
    
    # ペナルティ計算
    penalty = 0
    
    # Delisting警告ペナルティ
    delisting_count = len(analysis_results["delisting_warnings"])
    penalty += min(20, delisting_count * 2)
    
    # データ破損ペナルティ
    corruption_count = len(analysis_results["data_corruption_issues"])
    penalty += min(30, corruption_count * 10)
    
    # 完全性ペナルティ
    completeness_score = analysis_results["data_completeness"].get("overall_completeness_score", 1.0)
    penalty += (1.0 - completeness_score) * 25
    
    # 新鮮度ペナルティ
    freshness = analysis_results["quality_statistics"].get("data_freshness", {})
    freshness_score = freshness.get("freshness_score", 100)
    penalty += (100 - freshness_score) * 0.15
    
    final_score = max(0, base_score - penalty)
    return final_score

def count_source_types(data_sources: List[Dict[str, Any]]) -> Dict[str, int]:
    """データソースタイプの集計"""
    type_counts = {}
    for source in data_sources:
        source_type = source.get("source_type", "unknown")
        type_counts[source_type] = type_counts.get(source_type, 0) + 1
    return type_counts

def generate_data_quality_recommendations(analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """データ品質改善推奨事項の生成"""
    recommendations = []
    
    # Delisting警告対応
    delisting_count = len(analysis_results["delisting_warnings"])
    if delisting_count > 0:
        recommendations.append({
            "priority": "high",
            "category": "data_source",
            "title": "Delisting警告の対応",
            "description": f"{delisting_count}件のDelisting警告に対する代替データソースの確保",
            "effort": "medium",
            "impact": "high"
        })
    
    # データ破損対応
    corruption_count = len(analysis_results["data_corruption_issues"])
    if corruption_count > 0:
        recommendations.append({
            "priority": "high",
            "category": "data_integrity",
            "title": "データ破損の修復",
            "description": f"{corruption_count}ファイルのデータ破損問題の修復",
            "effort": "medium",
            "impact": "high"
        })
    
    # データ完全性向上
    completeness_score = analysis_results["data_completeness"].get("overall_completeness_score", 1.0)
    if completeness_score < 0.9:
        recommendations.append({
            "priority": "medium",
            "category": "data_completeness",
            "title": "データ完全性の向上",
            "description": f"データ完全性{completeness_score*100:.1f}%から90%以上への改善",
            "effort": "high",
            "impact": "medium"
        })
    
    return recommendations

def assess_data_health_status(overall_score: float) -> str:
    """データ健全性ステータスの評価"""
    if overall_score >= 80:
        return "healthy"
    elif overall_score >= 60:
        return "needs_attention"
    elif overall_score >= 40:
        return "poor"
    else:
        return "critical"

if __name__ == "__main__":
    try:
        results = analyze_data_quality()
        
        # 結果サマリー表示
        summary = results["summary_findings"]
        print(f"\n[CHART] データ品質分析サマリー:")
        print(f"データソース数: {summary['total_data_sources']}")
        print(f"Delisting警告: {summary['delisting_warnings_count']}")
        print(f"データ破損ファイル: {summary['data_corruption_files']}")
        print(f"全体品質スコア: {summary['overall_quality_score']:.1f}/100")
        print(f"データ完全性: {summary['data_completeness_score']*100:.1f}%")
        print(f"切替への影響: {summary['switching_impact_level']} ({summary['total_switching_impact_percent']:.1f}%)")
        print(f"データ健全性: {summary['data_health_status']}")
        
        if summary["primary_concerns"]:
            print(f"\n[WARNING] 主要懸念事項:")
            for concern in summary["primary_concerns"]:
                print(f"  - {concern}")
        
    except Exception as e:
        print(f"[ERROR] Task 5.4 エラー: {e}")
        import traceback
        traceback.print_exc()