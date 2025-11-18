"""
Scalable Data Lake Architecture for Breach Prediction System
Phase 1: Data Architecture Foundation

Provides hierarchical data storage with raw, processed, and enriched zones.
Supports parquet format for analytics and time/source-based partitioning.
"""

import os
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import logging
from enum import Enum

class DataZone(Enum):
    """Data lake zones for different data processing stages"""
    RAW = "raw"           # Original, unprocessed data
    PROCESSED = "processed"  # Cleaned and transformed data
    ENRICHED = "enriched"    # Enhanced with external data
    ANALYTICS = "analytics"  # Optimized for analysis/querying

class DataFormat(Enum):
    """Supported data formats"""
    PARQUET = "parquet"
    JSON = "json"
    CSV = "csv"

@dataclass
class DataLakeConfig:
    """Configuration for data lake operations"""
    base_path: str = "data_lake"
    compression: str = "snappy"
    partition_by: List[str] = None
    max_file_size_mb: int = 100
    retention_days: int = 365

    def __post_init__(self):
        if self.partition_by is None:
            self.partition_by = ["source", "year", "month"]

@dataclass
class DataMetadata:
    """Metadata for stored datasets"""
    dataset_id: str
    source: str
    data_type: str
    zone: DataZone
    format: DataFormat
    schema: Dict[str, str]
    record_count: int
    file_size_bytes: int
    partition_keys: List[str]
    created_at: datetime
    updated_at: datetime
    quality_score: float = 0.0
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

class DataLakeManager:
    """
    Manages hierarchical data lake with multiple zones and partitioning strategies
    """

    def __init__(self, config: DataLakeConfig = None):
        self.config = config or DataLakeConfig()
        self.base_path = Path(self.config.base_path)
        self.metadata_path = self.base_path / "metadata"
        self.logger = logging.getLogger(__name__)

        # Create directory structure
        self._create_directory_structure()

        # Initialize metadata store
        self.metadata_store = {}

    def _create_directory_structure(self):
        """Create the hierarchical directory structure"""
        for zone in DataZone:
            zone_path = self.base_path / zone.value
            zone_path.mkdir(parents=True, exist_ok=True)

            # Create subdirectories for common sources
            for source in ["external_threats", "internal_logs", "business_context", "industry_data"]:
                (zone_path / source).mkdir(exist_ok=True)

        # Metadata directory
        self.metadata_path.mkdir(exist_ok=True)

    def store_data(self,
                   df: pd.DataFrame,
                   dataset_id: str,
                   source: str,
                   data_type: str,
                   zone: DataZone = DataZone.RAW,
                   partition_keys: Dict[str, Any] = None,
                   metadata: Dict[str, Any] = None) -> str:
        """
        Store DataFrame in the data lake with partitioning

        Args:
            df: DataFrame to store
            dataset_id: Unique identifier for the dataset
            source: Data source name
            data_type: Type of data (breach_data, threat_intel, etc.)
            zone: Data lake zone
            partition_keys: Additional partitioning keys
            metadata: Additional metadata

        Returns:
            Path to stored file
        """
        try:
            # Prepare partitioning
            partition_info = self._prepare_partition_info(source, data_type, partition_keys)

            # Create file path
            file_path = self._create_partitioned_path(zone, partition_info, dataset_id)

            # Store data
            if self.config.compression:
                file_path = file_path.with_suffix(f".{self.config.compression}.parquet")
            else:
                file_path = file_path.with_suffix(".parquet")

            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to pyarrow table and save
            table = pa.Table.from_pandas(df)
            pq.write_table(table, file_path, compression=self.config.compression)

            # Create and store metadata
            file_size = file_path.stat().st_size
            data_metadata = DataMetadata(
                dataset_id=dataset_id,
                source=source,
                data_type=data_type,
                zone=zone,
                format=DataFormat.PARQUET,
                schema={col: str(dtype) for col, dtype in df.dtypes.items()},
                record_count=len(df),
                file_size_bytes=file_size,
                partition_keys=list(partition_info.keys()),
                created_at=datetime.now(),
                updated_at=datetime.now(),
                quality_score=metadata.get('quality_score', 0.0) if metadata else 0.0,
                tags=metadata.get('tags', []) if metadata else []
            )

            self._store_metadata(data_metadata)

            self.logger.info(f"Stored dataset {dataset_id} in {zone.value} zone: {file_path}")
            return str(file_path)

        except Exception as e:
            self.logger.error(f"Error storing data {dataset_id}: {str(e)}")
            raise

    def load_data(self,
                  dataset_id: str,
                  zone: DataZone = None,
                  filters: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Load data from the data lake

        Args:
            dataset_id: Dataset identifier
            zone: Specific zone to load from (searches all if None)
            filters: Partition filters

        Returns:
            Loaded DataFrame
        """
        try:
            # Find dataset metadata
            metadata = self._find_dataset_metadata(dataset_id, zone)
            if not metadata:
                raise FileNotFoundError(f"Dataset {dataset_id} not found")

            # Construct file path
            file_path = self._construct_file_path(metadata)

            if not file_path.exists():
                raise FileNotFoundError(f"Data file not found: {file_path}")

            # Load data
            table = pq.read_table(file_path)
            df = table.to_pandas()

            # Apply filters if provided
            if filters:
                for col, value in filters.items():
                    if col in df.columns:
                        df = df[df[col] == value]

            self.logger.info(f"Loaded dataset {dataset_id} with {len(df)} records")
            return df

        except Exception as e:
            self.logger.error(f"Error loading data {dataset_id}: {str(e)}")
            raise

    def query_data(self,
                   source: str = None,
                   data_type: str = None,
                   zone: DataZone = None,
                   date_from: datetime = None,
                   date_to: datetime = None,
                   filters: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Query data across partitions

        Args:
            source: Filter by source
            data_type: Filter by data type
            zone: Filter by zone
            date_from: Start date filter
            date_to: End date filter
            filters: Additional filters

        Returns:
            Combined DataFrame from matching partitions
        """
        try:
            matching_metadata = self._find_matching_metadata(
                source=source,
                data_type=data_type,
                zone=zone,
                date_from=date_from,
                date_to=date_to
            )

            if not matching_metadata:
                return pd.DataFrame()

            # Load and combine data
            dfs = []
            for metadata in matching_metadata:
                try:
                    df = self.load_data(metadata.dataset_id, metadata.zone, filters)
                    dfs.append(df)
                except Exception as e:
                    self.logger.warning(f"Failed to load {metadata.dataset_id}: {str(e)}")

            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                self.logger.info(f"Queried {len(combined_df)} records across {len(dfs)} partitions")
                return combined_df
            else:
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Error querying data: {str(e)}")
            raise

    def move_data_between_zones(self,
                               dataset_id: str,
                               from_zone: DataZone,
                               to_zone: DataZone,
                               transformation_func: callable = None) -> bool:
        """
        Move data between zones with optional transformation

        Args:
            dataset_id: Dataset to move
            from_zone: Source zone
            to_zone: Target zone
            transformation_func: Optional transformation function

        Returns:
            Success status
        """
        try:
            # Load data from source zone
            df = self.load_data(dataset_id, from_zone)

            # Apply transformation if provided
            if transformation_func:
                df = transformation_func(df)

            # Store in target zone
            metadata = self._find_dataset_metadata(dataset_id, from_zone)
            if metadata:
                self.store_data(
                    df=df,
                    dataset_id=dataset_id,
                    source=metadata.source,
                    data_type=metadata.data_type,
                    zone=to_zone,
                    metadata={"quality_score": metadata.quality_score, "tags": metadata.tags}
                )

                # Mark old metadata as moved
                metadata.tags.append(f"moved_to_{to_zone.value}")
                self._store_metadata(metadata)

                self.logger.info(f"Moved {dataset_id} from {from_zone.value} to {to_zone.value}")
                return True
            else:
                return False

        except Exception as e:
            self.logger.error(f"Error moving data {dataset_id}: {str(e)}")
            return False

    def cleanup_old_data(self, retention_days: int = None) -> int:
        """
        Clean up old data based on retention policy

        Args:
            retention_days: Override default retention period

        Returns:
            Number of files cleaned up
        """
        retention = retention_days or self.config.retention_days
        cutoff_date = datetime.now() - timedelta(days=retention)

        cleaned_count = 0
        try:
            for metadata_file in self.metadata_path.glob("*.json"):
                with open(metadata_file, 'r') as f:
                    metadata_dict = json.load(f)
                    metadata = DataMetadata(**metadata_dict)

                if metadata.created_at < cutoff_date:
                    # Remove data file
                    file_path = self._construct_file_path(metadata)
                    if file_path.exists():
                        file_path.unlink()

                    # Remove metadata file
                    metadata_file.unlink()

                    cleaned_count += 1

            self.logger.info(f"Cleaned up {cleaned_count} old data files")
            return cleaned_count

        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            return 0

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics for the data lake"""
        stats = {
            "total_datasets": 0,
            "total_size_bytes": 0,
            "datasets_by_zone": {},
            "datasets_by_source": {},
            "oldest_dataset": None,
            "newest_dataset": None
        }

        try:
            for metadata_file in self.metadata_path.glob("*.json"):
                with open(metadata_file, 'r') as f:
                    metadata_dict = json.load(f)
                    metadata = DataMetadata(**metadata_dict)

                stats["total_datasets"] += 1
                stats["total_size_bytes"] += metadata.file_size_bytes

                # Zone stats
                zone = metadata.zone.value
                if zone not in stats["datasets_by_zone"]:
                    stats["datasets_by_zone"][zone] = 0
                stats["datasets_by_zone"][zone] += 1

                # Source stats
                source = metadata.source
                if source not in stats["datasets_by_source"]:
                    stats["datasets_by_source"][source] = 0
                stats["datasets_by_source"][source] += 1

                # Date tracking
                if stats["oldest_dataset"] is None or metadata.created_at < stats["oldest_dataset"]:
                    stats["oldest_dataset"] = metadata.created_at
                if stats["newest_dataset"] is None or metadata.created_at > stats["newest_dataset"]:
                    stats["newest_dataset"] = metadata.created_at

            return stats

        except Exception as e:
            self.logger.error(f"Error getting storage stats: {str(e)}")
            return stats

    def _prepare_partition_info(self, source: str, data_type: str, partition_keys: Dict[str, Any] = None) -> Dict[str, Any]:
        """Prepare partitioning information"""
        partition_info = {
            "source": source,
            "data_type": data_type,
            "year": datetime.now().year,
            "month": datetime.now().month,
            "day": datetime.now().day
        }

        if partition_keys:
            partition_info.update(partition_keys)

        return partition_info

    def _create_partitioned_path(self, zone: DataZone, partition_info: Dict[str, Any], dataset_id: str) -> Path:
        """Create partitioned file path"""
        path_parts = [self.base_path / zone.value]

        # Add partition directories
        for key in self.config.partition_by:
            if key in partition_info:
                path_parts.append(str(partition_info[key]))

        path_parts.append(f"{dataset_id}")
        return Path(*path_parts)

    def _store_metadata(self, metadata: DataMetadata):
        """Store dataset metadata"""
        metadata_file = self.metadata_path / f"{metadata.dataset_id}.json"
        with open(metadata_file, 'w') as f:
            json.dump(asdict(metadata), f, indent=2, default=str)

        self.metadata_store[metadata.dataset_id] = metadata

    def _find_dataset_metadata(self, dataset_id: str, zone: DataZone = None) -> Optional[DataMetadata]:
        """Find metadata for a dataset"""
        metadata_file = self.metadata_path / f"{dataset_id}.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata_dict = json.load(f)
                metadata = DataMetadata(**metadata_dict)

                if zone is None or metadata.zone == zone:
                    return metadata

        return None

    def _find_matching_metadata(self, source: str = None, data_type: str = None,
                               zone: DataZone = None, date_from: datetime = None,
                               date_to: datetime = None) -> List[DataMetadata]:
        """Find metadata matching criteria"""
        matching = []

        for metadata_file in self.metadata_path.glob("*.json"):
            with open(metadata_file, 'r') as f:
                metadata_dict = json.load(f)
                metadata = DataMetadata(**metadata_dict)

            # Apply filters
            if source and metadata.source != source:
                continue
            if data_type and metadata.data_type != data_type:
                continue
            if zone and metadata.zone != zone:
                continue
            if date_from and metadata.created_at < date_from:
                continue
            if date_to and metadata.created_at > date_to:
                continue

            matching.append(metadata)

        return matching

    def _construct_file_path(self, metadata: DataMetadata) -> Path:
        """Construct file path from metadata"""
        partition_info = {
            "source": metadata.source,
            "data_type": metadata.data_type,
            "year": metadata.created_at.year,
            "month": metadata.created_at.month,
            "day": metadata.created_at.day
        }

        path = self._create_partitioned_path(metadata.zone, partition_info, metadata.dataset_id)

        if self.config.compression:
            return path.with_suffix(f".{self.config.compression}.parquet")
        else:
            return path.with_suffix(".parquet")

# Convenience functions for common operations
def create_data_lake_manager(config: DataLakeConfig = None) -> DataLakeManager:
    """Create and initialize a data lake manager"""
    return DataLakeManager(config)

def store_dataset(df: pd.DataFrame, dataset_id: str, source: str, data_type: str,
                 zone: DataZone = DataZone.RAW, **kwargs) -> str:
    """Convenience function to store a dataset"""
    manager = DataLakeManager()
    return manager.store_data(df, dataset_id, source, data_type, zone, **kwargs)

def load_dataset(dataset_id: str, zone: DataZone = None, **kwargs) -> pd.DataFrame:
    """Convenience function to load a dataset"""
    manager = DataLakeManager()
    return manager.load_data(dataset_id, zone, **kwargs)

if __name__ == "__main__":
    # Example usage
    manager = DataLakeManager()

    # Create sample data
    sample_data = pd.DataFrame({
        'company': ['TechCorp', 'FinanceInc', 'HealthSys'],
        'breach_date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'records_affected': [1000, 5000, 2500],
        'risk_score': [0.3, 0.7, 0.5]
    })

    # Store data
    file_path = manager.store_data(
        df=sample_data,
        dataset_id="sample_breaches_2024",
        source="hibp_api",
        data_type="breach_data",
        zone=DataZone.RAW
    )

    print(f"Stored data at: {file_path}")

    # Load data
    loaded_data = manager.load_data("sample_breaches_2024", DataZone.RAW)
    print(f"Loaded {len(loaded_data)} records")

    # Get storage stats
    stats = manager.get_storage_stats()
    print(f"Data lake stats: {stats}")
