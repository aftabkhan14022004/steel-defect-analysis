CREATE DATABASE IF NOT EXISTS steel_defects;
USE steel_defects;

CREATE TABLE production_batches (
    batch_id VARCHAR(10) PRIMARY KEY,
    timestamp VARCHAR(20),
    shift ENUM('morning','evening','night'),
    furnace_temp DECIMAL(6,2),
    rolling_speed DECIMAL(5,2)
);

CREATE TABLE machine_parameters (
    batch_id VARCHAR(10) PRIMARY KEY,
    operator_id VARCHAR(5),
    machine_id VARCHAR(3),
    FOREIGN KEY (batch_id) REFERENCES production_batches(batch_id)
);

CREATE TABLE defect_inspections (
    inspection_id INT AUTO_INCREMENT PRIMARY KEY,
    batch_id VARCHAR(10),
    defect_type VARCHAR(20),
    defect_count INT,
    FOREIGN KEY (batch_id) REFERENCES production_batches(batch_id)
);