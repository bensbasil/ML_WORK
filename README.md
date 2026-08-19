                    config.py
                        │
                        ▼
                 run_pipeline.py
                        │
                        ▼
            processing_pipeline.py
             /        |         \
            ▼         ▼          ▼
     ingestion.py validation.py extractor.py
                                      │
                                      ▼
                              tile_features.py
                                      │
                                      ▼
                              Flat dictionaries
                                      │
                                      ▼
                                Batch rows
                                      │
                                      ▼
                                  DataFrame
                                      │
                                      ▼
                             Parquet part files