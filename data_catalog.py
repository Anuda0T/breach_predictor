"""
Data Catalog & Metadata Management System
Phase 1: Data Architecture Foundation

Provides comprehensive metadata management, data discovery, and lineage tracking
for the breach prediction data lake.
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
import logging
from enum import Enum
import re

class DataSensitivity(Enum):
    """Data sensitivity levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

class DataQuality(Enum):
    """Data quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

@dataclass
class DataLineage:
    """Data lineage information"""
    dataset_id: str
    parent_datasets: List[str]
    transformation_steps: List[str]
    created_by: str
    created_at: datetime
    version: str = "1.0"

@dataclass
class DataAsset:
    """Comprehensive data asset metadata"""
    asset_id: str
    name: str
    description: str
    dataset_id: str
    source_system: str
    data_type: str
    format: str
    schema: Dict[str, Any]
    record_count: int
    file_size_bytes: int
    created_at: datetime
    updated_at: datetime
    owner: str
    tags: List[str]
    sensitivity: DataSensitivity
    quality_score: float
    quality_level: DataQuality
    retention_policy: str
    access_permissions: Dict[str, List[str]]
    data_lineage: DataLineage
    statistics: Dict[str, Any] = None
    sample_data: List[Dict[str, Any]] = None
    validation_rules: List[str] = None

    def __post_init__(self):
        if self.statistics is None:
            self.statistics = {}
        if self.sample_data is None:
            self.sample_data = []
        if self.validation_rules is None:
            self.validation_rules = []

class DataCatalog:
    """
    Comprehensive data catalog with metadata management and discovery capabilities
    """

    def __init__(self, db_path: str = "data_catalog.db"):
        self.db_path = Path(db_path)
        self.logger = logging.getLogger(__name__)
        self._initialize_database()

    def _initialize_database(self):
        """Initialize SQLite database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Data assets table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS data_assets (
                    asset_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    dataset_id TEXT NOT NULL,
                    source_system TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    format TEXT NOT NULL,
                    schema_json TEXT,
                    record_count INTEGER,
                    file_size_bytes INTEGER,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    owner TEXT,
                    tags_json TEXT,
                    sensitivity TEXT,
                    quality_score REAL,
                    quality_level TEXT,
                    retention_policy TEXT,
                    access_permissions_json TEXT,
                    data_lineage_json TEXT,
                    statistics_json TEXT,
                    sample_data_json TEXT,
                    validation_rules_json TEXT
                )
            ''')

            # Search index table for fast text search
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS search_index (
                    asset_id TEXT,
                    search_text TEXT,
                    FOREIGN KEY (asset_id) REFERENCES data_assets (asset_id)
                )
            ''')

            # Tags table for efficient tag-based queries
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS asset_tags (
                    asset_id TEXT,
                    tag TEXT,
                    FOREIGN KEY (asset_id) REFERENCES data_assets (asset_id)
                )
            ''')

            # Data lineage table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS data_lineage (
                    dataset_id TEXT PRIMARY KEY,
                    parent_datasets_json TEXT,
                    transformation_steps_json TEXT,
                    created_by TEXT,
                    created_at TEXT,
                    version TEXT
                )
            ''')

            conn.commit()

    def register_asset(self, asset: DataAsset) -> bool:
        """
        Register a new data asset in the catalog

        Args:
            asset: DataAsset object to register

        Returns:
            Success status
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Insert main asset record
                cursor.execute('''
                    INSERT OR REPLACE INTO data_assets
                    (asset_id, name, description, dataset_id, source_system, data_type, format,
                     schema_json, record_count, file_size_bytes, created_at, updated_at, owner,
                     tags_json, sensitivity, quality_score, quality_level, retention_policy,
                     access_permissions_json, data_lineage_json, statistics_json,
                     sample_data_json, validation_rules_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    asset.asset_id, asset.name, asset.description, asset.dataset_id,
                    asset.source_system, asset.data_type, asset.format,
                    json.dumps(asset.schema), asset.record_count, asset.file_size_bytes,
                    asset.created_at.isoformat(), asset.updated_at.isoformat(), asset.owner,
                    json.dumps(asset.tags), asset.sensitivity.value, asset.quality_score,
                    asset.quality_level.value, asset.retention_policy,
                    json.dumps(asset.access_permissions), json.dumps(asdict(asset.data_lineage)),
                    json.dumps(asset.statistics), json.dumps(asset.sample_data),
                    json.dumps(asset.validation_rules)
                ))

                # Insert search index
                search_text = self._create_search_text(asset)
                cursor.execute('INSERT INTO search_index (asset_id, search_text) VALUES (?, ?)',
                             (asset.asset_id, search_text))

                # Insert tags
                for tag in asset.tags:
                    cursor.execute('INSERT INTO asset_tags (asset_id, tag) VALUES (?, ?)',
                                 (asset.asset_id, tag))

                # Insert lineage
                cursor.execute('''
                    INSERT OR REPLACE INTO data_lineage
                    (dataset_id, parent_datasets_json, transformation_steps_json,
                     created_by, created_at, version)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    asset.data_lineage.dataset_id,
                    json.dumps(asset.data_lineage.parent_datasets),
                    json.dumps(asset.data_lineage.transformation_steps),
                    asset.data_lineage.created_by,
                    asset.data_lineage.created_at.isoformat(),
                    asset.data_lineage.version
                ))

                conn.commit()

            self.logger.info(f"Registered data asset: {asset.asset_id}")
            return True

        except Exception as e:
            self.logger.error(f"Error registering asset {asset.asset_id}: {str(e)}")
            return False

    def search_assets(self,
                     query: str = None,
                     data_type: str = None,
                     source_system: str = None,
                     tags: List[str] = None,
                     sensitivity: DataSensitivity = None,
                     quality_min: float = None,
                     limit: int = 50) -> List[DataAsset]:
        """
        Search for data assets based on various criteria

        Args:
            query: Text search query
            data_type: Filter by data type
            source_system: Filter by source system
            tags: Filter by tags (must have all specified tags)
            sensitivity: Filter by sensitivity level
            quality_min: Minimum quality score
            limit: Maximum number of results

        Returns:
            List of matching DataAsset objects
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Build query
                conditions = []
                params = []

                if query:
                    conditions.append("search_text LIKE ?")
                    params.append(f"%{query}%")

                if data_type:
                    conditions.append("data_type = ?")
                    params.append(data_type)

                if source_system:
                    conditions.append("source_system = ?")
                    params.append(source_system)

                if sensitivity:
                    conditions.append("sensitivity = ?")
                    params.append(sensitivity.value)

                if quality_min is not None:
                    conditions.append("quality_score >= ?")
                    params.append(quality_min)

                # Handle tag filtering
                if tags:
                    tag_conditions = " AND ".join(["?" for _ in tags])
                    conditions.append(f"asset_id IN (SELECT asset_id FROM asset_tags WHERE tag IN ({tag_conditions}))")
                    params.extend(tags)

                where_clause = " AND ".join(conditions) if conditions else "1=1"

                sql = f'''
                    SELECT * FROM data_assets
                    WHERE {where_clause}
                    ORDER BY updated_at DESC
                    LIMIT ?
                '''
                params.append(limit)

                cursor.execute(sql, params)
                rows = cursor.fetchall()

                assets = []
                for row in rows:
                    asset = self._row_to_asset(row)
                    assets.append(asset)

                return assets

        except Exception as e:
            self.logger.error(f"Error searching assets: {str(e)}")
            return []

    def get_asset(self, asset_id: str) -> Optional[DataAsset]:
        """
        Get a specific data asset by ID

        Args:
            asset_id: Asset identifier

        Returns:
            DataAsset object or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM data_assets WHERE asset_id = ?", (asset_id,))
                row = cursor.fetchone()

                if row:
                    return self._row_to_asset(row)

        except Exception as e:
            self.logger.error(f"Error getting asset {asset_id}: {str(e)}")

        return None

    def update_asset(self, asset_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update asset metadata

        Args:
            asset_id: Asset to update
            updates: Dictionary of fields to update

        Returns:
            Success status
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Build update query
                set_parts = []
                params = []

                for field, value in updates.items():
                    if field in ['schema', 'tags', 'access_permissions', 'statistics',
                               'sample_data', 'validation_rules']:
                        set_parts.append(f"{field}_json = ?")
                        params.append(json.dumps(value))
                    elif field in ['created_at', 'updated_at']:
                        set_parts.append(f"{field} = ?")
                        params.append(value.isoformat())
                    elif field == 'sensitivity' and isinstance(value, DataSensitivity):
                        set_parts.append(f"{field} = ?")
                        params.append(value.value)
                    elif field == 'quality_level' and isinstance(value, DataQuality):
                        set_parts.append(f"{field} = ?")
                        params.append(value.value)
                    else:
                        set_parts.append(f"{field} = ?")
                        params.append(value)

                if set_parts:
                    set_clause = ", ".join(set_parts)
                    sql = f"UPDATE data_assets SET {set_clause}, updated_at = ? WHERE asset_id = ?"
                    params.extend([datetime.now().isoformat(), asset_id])

                    cursor.execute(sql, params)

                    # Update search index if name or description changed
                    if 'name' in updates or 'description' in updates:
                        asset = self.get_asset(asset_id)
                        if asset:
                            search_text = self._create_search_text(asset)
                            cursor.execute("UPDATE search_index SET search_text = ? WHERE asset_id = ?",
                                         (search_text, asset_id))

                    conn.commit()

            self.logger.info(f"Updated asset: {asset_id}")
            return True

        except Exception as e:
            self.logger.error(f"Error updating asset {asset_id}: {str(e)}")
            return False

    def delete_asset(self, asset_id: str) -> bool:
        """
        Delete a data asset from the catalog

        Args:
            asset_id: Asset to delete

        Returns:
            Success status
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Delete from all tables
                tables = ['data_assets', 'search_index', 'asset_tags']
                for table in tables:
                    cursor.execute(f"DELETE FROM {table} WHERE asset_id = ?", (asset_id,))

                conn.commit()

            self.logger.info(f"Deleted asset: {asset_id}")
            return True

        except Exception as e:
            self.logger.error(f"Error deleting asset {asset_id}: {str(e)}")
            return False

    def get_data_lineage(self, dataset_id: str) -> Optional[DataLineage]:
        """
        Get data lineage information for a dataset

        Args:
            dataset_id: Dataset identifier

        Returns:
            DataLineage object or None
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM data_lineage WHERE dataset_id = ?", (dataset_id,))
                row = cursor.fetchone()

                if row:
                    return DataLineage(
                        dataset_id=row[0],
                        parent_datasets=json.loads(row[1]),
                        transformation_steps=json.loads(row[2]),
                        created_by=row[3],
                        created_at=datetime.fromisoformat(row[4]),
                        version=row[5]
                    )

        except Exception as e:
            self.logger.error(f"Error getting lineage for {dataset_id}: {str(e)}")

        return None

    def get_catalog_stats(self) -> Dict[str, Any]:
        """Get catalog statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                stats = {}

                # Total assets
                cursor.execute("SELECT COUNT(*) FROM data_assets")
                stats['total_assets'] = cursor.fetchone()[0]

                # Assets by data type
                cursor.execute("SELECT data_type, COUNT(*) FROM data_assets GROUP BY data_type")
                stats['assets_by_type'] = dict(cursor.fetchall())

                # Assets by source system
                cursor.execute("SELECT source_system, COUNT(*) FROM data_assets GROUP BY source_system")
                stats['assets_by_source'] = dict(cursor.fetchall())

                # Assets by sensitivity
                cursor.execute("SELECT sensitivity, COUNT(*) FROM data_assets GROUP BY sensitivity")
                stats['assets_by_sensitivity'] = dict(cursor.fetchall())

                # Average quality score
                cursor.execute("SELECT AVG(quality_score) FROM data_assets")
                stats['avg_quality_score'] = cursor.fetchone()[0] or 0

                # Total data size
                cursor.execute("SELECT SUM(file_size_bytes) FROM data_assets")
                stats['total_data_size_bytes'] = cursor.fetchone()[0] or 0

                return stats

        except Exception as e:
            self.logger.error(f"Error getting catalog stats: {str(e)}")
            return {}

    def discover_related_assets(self, asset_id: str, max_depth: int = 2) -> List[DataAsset]:
        """
        Discover related assets based on lineage and tags

        Args:
            asset_id: Starting asset ID
            max_depth: Maximum relationship depth

        Returns:
            List of related assets
        """
        try:
            asset = self.get_asset(asset_id)
            if not asset:
                return []

            related_ids = set()

            # Find assets with same tags
            if asset.tags:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    tag_placeholders = ",".join(["?" for _ in asset.tags])
                    cursor.execute(f'''
                        SELECT DISTINCT asset_id FROM asset_tags
                        WHERE tag IN ({tag_placeholders}) AND asset_id != ?
                    ''', asset.tags + [asset_id])

                    related_ids.update([row[0] for row in cursor.fetchall()])

            # Find assets from same source system
            related_assets = self.search_assets(
                source_system=asset.source_system,
                limit=20
            )
            related_ids.update([a.asset_id for a in related_assets if a.asset_id != asset_id])

            # Convert to asset objects
            related_assets = []
            for rid in list(related_ids)[:10]:  # Limit to 10
                asset_obj = self.get_asset(rid)
                if asset_obj:
                    related_assets.append(asset_obj)

            return related_assets

        except Exception as e:
            self.logger.error(f"Error discovering related assets for {asset_id}: {str(e)}")
            return []

    def _row_to_asset(self, row) -> DataAsset:
        """Convert database row to DataAsset object"""
        return DataAsset(
            asset_id=row[0],
            name=row[1],
            description=row[2],
            dataset_id=row[3],
            source_system=row[4],
            data_type=row[5],
            format=row[6],
            schema=json.loads(row[7]) if row[7] else {},
            record_count=row[8],
            file_size_bytes=row[9],
            created_at=datetime.fromisoformat(row[10]),
            updated_at=datetime.fromisoformat(row[11]),
            owner=row[12],
            tags=json.loads(row[13]) if row[13] else [],
            sensitivity=DataSensitivity(row[14]),
            quality_score=row[15],
            quality_level=DataQuality(row[16]),
            retention_policy=row[17],
            access_permissions=json.loads(row[18]) if row[18] else {},
            data_lineage=DataLineage(**json.loads(row[19])),
            statistics=json.loads(row[20]) if row[20] else {},
            sample_data=json.loads(row[21]) if row[21] else [],
            validation_rules=json.loads(row[22]) if row[22] else []
        )

    def _create_search_text(self, asset: DataAsset) -> str:
        """Create searchable text from asset metadata"""
        search_components = [
            asset.name,
            asset.description or "",
            asset.dataset_id,
            asset.source_system,
            asset.data_type,
            asset.owner or "",
            " ".join(asset.tags)
        ]

        # Add schema field names
        if asset.schema:
            search_components.extend(asset.schema.keys())

        return " ".join(search_components).lower()

# Convenience functions
def create_data_catalog(db_path: str = "data_catalog.db") -> DataCatalog:
    """Create and initialize a data catalog"""
    return DataCatalog(db_path)

def register_data_asset(asset: DataAsset, db_path: str = "data_catalog.db") -> bool:
    """Convenience function to register a data asset"""
    catalog = DataCatalog(db_path)
    return catalog.register_asset(asset)

def search_data_assets(query: str = None, **filters) -> List[DataAsset]:
    """Convenience function to search data assets"""
    catalog = DataCatalog()
    return catalog.search_assets(query=query, **filters)

if __name__ == "__main__":
    # Example usage
    catalog = DataCatalog()

    # Create sample asset
    lineage = DataLineage(
        dataset_id="sample_breaches",
        parent_datasets=[],
        transformation_steps=["collected from HIBP API"],
        created_by="data_collector",
        created_at=datetime.now()
    )

    asset = DataAsset(
        asset_id="hibp_breaches_2024",
        name="HIBP Breach Data 2024",
        description="Breach data collected from Have I Been Pwned API",
        dataset_id="hibp_breaches_2024",
        source_system="hibp_api",
        data_type="breach_data",
        format="parquet",
        schema={"company": "string", "breach_date": "date", "records_affected": "int64"},
        record_count=500,
        file_size_bytes=1024000,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        owner="data_team",
        tags=["breach", "security", "external"],
        sensitivity=DataSensitivity.CONFIDENTIAL,
        quality_score=0.85,
        quality_level=DataQuality.GOOD,
        retention_policy="7_years",
        access_permissions={"read": ["analyst", "admin"], "write": ["admin"]},
        data_lineage=lineage
    )

    # Register asset
    success = catalog.register_asset(asset)
    print(f"Asset registration: {'Success' if success else 'Failed'}")

    # Search assets
    results = catalog.search_assets(query="breach")
    print(f"Found {len(results)} assets matching 'breach'")

    # Get catalog stats
    stats = catalog.get_catalog_stats()
    print(f"Catalog stats: {stats}")
