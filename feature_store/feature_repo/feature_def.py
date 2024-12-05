from datetime import timedelta

import pandas as pd

from feast import (
    Entity,
    FeatureService,
    FeatureView,
    Field,
    FileSource,
    Project,
    PushSource,
    RequestSource,
    ValueType
)
from feast.feature_logging import LoggingConfig
from feast.infra.offline_stores.file_source import FileLoggingDestination
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float32, Float64, Int64


# Define a project 
project = Project(name="feature_store", description="A project for misuse detection in containers using classification algorithms.")



# File source 
misuse_det_file_source = FileSource(
    path="data/predictors.parquet",
    timestamp_field='event_timestamp',
    # created_timestamp_column='event_timestamp',
)


# Define an entity for the feature store
feature_entity = Entity(
    name="misuse_det_entity",
    value_type=ValueType.FLOAT,
    join_keys=['feature_id'],
    description="A entity for misuse detection in containers using classification algorithms."
)


# Define a feature view for the feature store
misuse_det_feature_view = FeatureView(
    name="misuse_det_feature_view",
    entities=[feature_entity],
    ttl=timedelta(seconds=86400 * 1), # 1 day
    schema=[
        Field(name="PC1", dtype=Float64, description="Feature 1"),
        Field(name="PC2", dtype=Float64, description="Feature 2"),
        Field(name="PC3", dtype=Float64, description="Feature 3"),
        Field(name="PC4", dtype=Float64, description="Feature 4"),
        Field(name="PC5", dtype=Float64, description="Feature 5"),
        Field(name="PC6", dtype=Float64, description="Feature 6"),
        Field(name="PC7", dtype=Float64, description="Feature 7"),
        Field(name="PC8", dtype=Float64, description="Feature 8"),
        Field(name="PC9", dtype=Float64, description="Feature 9"),
    ],
    online=True,
    source=misuse_det_file_source,
    tags={"release": "final"},
)