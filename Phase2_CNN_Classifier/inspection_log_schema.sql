-- Schema for the inspection_log table used by the Streamlit app's
-- MySQL logging feature (Phase2_CNN_Classifier/model_utils.py, log_inspection()).
-- Standalone table -- no foreign key to Phase 1's production_batches.
-- batch_id here is free-text entered per inspection, not a validated
-- reference to a real production batch. See README Known Limitations.

CREATE TABLE inspection_log (
    inspection_id INT AUTO_INCREMENT PRIMARY KEY,
    batch_id VARCHAR(10),
    image_file VARCHAR(100),
    actual_defect VARCHAR(20),
    predicted_defect VARCHAR(20),
    confidence DECIMAL(5,2),
    decision VARCHAR(30),
    line_number VARCHAR(10),
    inspection_time DATETIME
);
