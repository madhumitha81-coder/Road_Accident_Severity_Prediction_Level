TARGET_COLUMN = "accident_severity"

BASE_DROP_COLUMNS = [
    "age_band_of_driver",
    "time",
    "vehicle_driver_relation",
    "work_of_casuality",
    "fitness_of_casuality",
    "day_of_week",
    "casualty_severity",
    "sex_of_driver",
    "educational_level",
    "defect_of_vehicle",
    "owner_of_vehicle",
    "service_year_of_vehicle",
    "road_surface_type",
    "sex_of_casualty",
    "age_band_of_casualty",
    "number_of_casualties",
]

REQUESTED_EXCLUDED_COLUMNS = [
    "area_accident_occured",
    "casualty_class",
    "cause_of_accident",
    "type_of_collision",
]


def get_training_drop_columns():
    ordered_columns = []
    seen_columns = set()

    for column in [*BASE_DROP_COLUMNS, *REQUESTED_EXCLUDED_COLUMNS]:
        if column == TARGET_COLUMN or column in seen_columns:
            continue

        ordered_columns.append(column)
        seen_columns.add(column)

    return ordered_columns


def humanize_column_name(column_name: str) -> str:
    return " ".join(part.capitalize() for part in column_name.split("_"))
