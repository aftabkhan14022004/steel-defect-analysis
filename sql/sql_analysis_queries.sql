
"""QUERY 1"""

WITH clean_batches AS (
    SELECT batch_id,
       CASE
         WHEN length(timestamp)= 16 then STR_TO_DATE(TIMESTAMP, '%Y-%m-%d %H:%i')
         else STR_TO_DATE(TIMESTAMP,'%Y-%m-%d %H:%i:%s')
       END AS clean_ts,
       shift,
       furnace_temp,
       rolling_speed
    FROM production_batches
    )
SELECT * from clean_batches limit 10



"""QUERY 2"""



with shift_data as (
    select
         p.shift,
         p.batch_id,
         d.defect_count
    from production_batches p
    join defect_inspections d ON p.batch_id =d.batch_id
)
SELECT
      shift,
      count(DISTINCT batch_id) as total_batches,
      sum(defect_count) as total_defects,
      round(avg(defect_count),2) as avg_defects_per_inspection
from shift_data
group by shift;





"""QUERY 3"""

SELECT
     p.batch_id,
     p.furnace_temp,
     p.rolling_speed,
     m.machine_id,
     m.operator_id,
     SUM(d.defect_count) AS total_defects
FROM production_batches p
JOIN defect_inspections d ON p.batch_id=d.batch_id
JOIN machine_parameters m on p.batch_id=m.batch_id
group BY p.batch_id,p.furnace_temp,p.rolling_speed,m.machine_id,m.operator_id
HAVING sum(d.defect_count) > (
    SELECT AVG(batch_total)
    from (
        SELECT sum(defect_count) as batch_total
        from defect_inspections
        group by batch_id
     ) as sub
  )
  order by total_defects DESC


