-- Add training_start_date and rename training_date to training_cutoff_date
-- This makes it explicit what data range was used for training

-- Add training_start_date column
ALTER TABLE commodity.forecast.trained_models
ADD COLUMN training_start_date DATE;

-- Rename training_date to training_cutoff_date
ALTER TABLE commodity.forecast.trained_models
RENAME COLUMN training_date TO training_cutoff_date;

-- Verify the changes
DESCRIBE TABLE commodity.forecast.trained_models;
