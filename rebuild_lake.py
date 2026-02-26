import json
import logging
import re
import sqlite3
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("rebuild_lake")

def deep_get(d: dict, dotpath: str, default: Any = None) -> Any:
    keys = dotpath.split(".")
    curr = d
    for k in keys:
        if isinstance(curr, dict) and k in curr:
            curr = curr[k]
        else:
            return default
    return curr

def rebuild():
    results_dir = Path("results")
    db_path = Path("idea_lake.db")
    
    # 1. Connect to DB and ensure schema
    import sys
    sys.path.append("orze")
    from idea_lake import IdeaLake
    lake = IdeaLake(str(db_path))
    
    # 2. Scan all idea directories
    idea_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("idea-")])
    logger.info(f"Found {len(idea_dirs)} idea directories in {results_dir}")
    
    count = 0
    for d in idea_dirs:
        idea_id = d.name
        metrics_path = d / "metrics.json"
        config_path = d / "resolved_config.yaml"
        
        if not metrics_path.exists():
            continue
            
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            config_yaml = ""
            if config_path.exists():
                config_yaml = config_path.read_text(encoding="utf-8")
            
            title = metrics.get("idea_title") or idea_id
            status = metrics.get("status", "archived").lower()
            
            # Extract eval metrics for report
            eval_metrics = {}
            # Try to find common report metrics
            for report_file in ["fedex_test_report.json", "nexar_test_report.json"]:
                report_path = d / report_file
                if report_path.exists():
                    try:
                        rd = json.loads(report_path.read_text(encoding="utf-8"))
                        rm = rd.get("metrics", {})
                        for k, v in rm.items():
                            key = f"{report_file.split('_')[0]}_{k}"
                            eval_metrics[key] = v
                    except:
                        pass
            
            # If no report metrics found, try to pull from metrics.json test_metrics
            if not eval_metrics and "test_metrics" in metrics:
                for k, v in metrics["test_metrics"].items():
                    eval_metrics[f"test_{k}"] = v

            lake.insert(
                idea_id=idea_id,
                title=title,
                config_yaml=config_yaml,
                raw_markdown="",
                eval_metrics=eval_metrics,
                status=status,
                created_at=metrics.get("timestamp")
            )
            count += 1
            if count % 1000 == 0:
                logger.info(f"Processed {count} ideas...")
        except Exception as e:
            logger.warning(f"Failed to process {idea_id}: {e}")
            
    lake.close()
    logger.info(f"Rebuild complete. Total {count} ideas indexed.")

if __name__ == "__main__":
    rebuild()
