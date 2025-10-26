"""
Database operations for M&S Reduced Items Prediction System.
SQLite wrapper with schema for detections, videos, and model metrics.
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path


class DatabaseManager:
    """SQLite database manager for the M&S prediction system."""
    
    def __init__(self, db_path: str = "data/detections.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Store each detection instance
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT NOT NULL,
                    frame_number INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    product_id TEXT,
                    product_name TEXT,
                    product_category TEXT,
                    confidence FLOAT,
                    location_branch TEXT,
                    day_of_week INTEGER,
                    hour_of_day INTEGER,
                    date DATE,
                    sticker_bbox TEXT,
                    product_bbox TEXT,
                    frame_path TEXT
                )
            """)
            
            # Track video uploads
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS videos (
                    id TEXT PRIMARY KEY,
                    upload_date DATETIME,
                    branch_location TEXT,
                    contributor_id TEXT,
                    processed BOOLEAN DEFAULT FALSE,
                    frame_count INTEGER
                )
            """)
            
            # Store prediction model performance
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_version TEXT,
                    train_date DATETIME,
                    accuracy FLOAT,
                    precision_recall JSON,
                    feature_importance JSON
                )
            """)
            
            conn.commit()
    
    def add_detection(self, 
                     video_id: str,
                     frame_number: int,
                     product_id: Optional[str] = None,
                     product_name: Optional[str] = None,
                     product_category: Optional[str] = None,
                     confidence: Optional[float] = None,
                     location_branch: Optional[str] = None,
                     day_of_week: Optional[int] = None,
                     hour_of_day: Optional[int] = None,
                     date: Optional[str] = None,
                     sticker_bbox: Optional[Dict] = None,
                     product_bbox: Optional[Dict] = None,
                     frame_path: Optional[str] = None) -> int:
        """Add a new detection record."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO detections (
                    video_id, frame_number, product_id, product_name, 
                    product_category, confidence, location_branch,
                    day_of_week, hour_of_day, date, sticker_bbox,
                    product_bbox, frame_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                video_id, frame_number, product_id, product_name,
                product_category, confidence, location_branch,
                day_of_week, hour_of_day, date,
                json.dumps(sticker_bbox) if sticker_bbox else None,
                json.dumps(product_bbox) if product_bbox else None,
                frame_path
            ))
            conn.commit()
            return cursor.lastrowid
    
    def add_video(self, 
                video_id: str,
                branch_location: str,
                contributor_id: str,
                frame_count: int = 0) -> None:
        """Add a new video record."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO videos (
                    id, upload_date, branch_location, contributor_id, frame_count
                ) VALUES (?, ?, ?, ?, ?)
            """, (video_id, datetime.now(), branch_location, contributor_id, frame_count))
            conn.commit()
    
    def mark_video_processed(self, video_id: str) -> None:
        """Mark a video as processed."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE videos SET processed = TRUE WHERE id = ?
            """, (video_id,))
            conn.commit()
    
    def get_detections(self, 
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      product_category: Optional[str] = None,
                      location_branch: Optional[str] = None,
                      limit: int = 1000) -> List[Dict]:
        """Query detections with optional filters."""
        query = "SELECT * FROM detections WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        if product_category:
            query += " AND product_category = ?"
            params.append(product_category)
        if location_branch:
            query += " AND location_branch = ?"
            params.append(location_branch)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def get_training_data(self, 
                         start_date: str,
                         end_date: str,
                         min_samples: int = 100) -> List[Dict]:
        """Get historical data for model training."""
        query = """
            SELECT 
                product_id, product_name, product_category,
                day_of_week, hour_of_day, date, location_branch,
                COUNT(*) as detection_count
            FROM detections 
            WHERE date BETWEEN ? AND ?
            GROUP BY product_id, product_name, product_category, 
                     day_of_week, hour_of_day, date, location_branch
            HAVING COUNT(*) >= ?
            ORDER BY date DESC
        """
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, (start_date, end_date, min_samples))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def get_video_status(self, video_id: str) -> Optional[Dict]:
        """Get video processing status."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM videos WHERE id = ?", (video_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def add_model_metrics(self,
                         model_version: str,
                         accuracy: float,
                         precision_recall: Dict,
                         feature_importance: Dict) -> None:
        """Add model performance metrics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO model_metrics (
                    model_version, train_date, accuracy, 
                    precision_recall, feature_importance
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                model_version, datetime.now(), accuracy,
                json.dumps(precision_recall),
                json.dumps(feature_importance)
            ))
            conn.commit()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total detections
            cursor.execute("SELECT COUNT(*) FROM detections")
            total_detections = cursor.fetchone()[0]
            
            # Total videos
            cursor.execute("SELECT COUNT(*) FROM videos")
            total_videos = cursor.fetchone()[0]
            
            # Processed videos
            cursor.execute("SELECT COUNT(*) FROM videos WHERE processed = TRUE")
            processed_videos = cursor.fetchone()[0]
            
            # Unique products detected
            cursor.execute("SELECT COUNT(DISTINCT product_id) FROM detections WHERE product_id IS NOT NULL")
            unique_products = cursor.fetchone()[0]
            
            # Detections by category
            cursor.execute("""
                SELECT product_category, COUNT(*) as count 
                FROM detections 
                WHERE product_category IS NOT NULL 
                GROUP BY product_category 
                ORDER BY count DESC
            """)
            category_stats = dict(cursor.fetchall())
            
            return {
                "total_detections": total_detections,
                "total_videos": total_videos,
                "processed_videos": processed_videos,
                "unique_products": unique_products,
                "category_stats": category_stats
            }
