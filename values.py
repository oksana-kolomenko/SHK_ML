from enum import Enum


class Classification(Enum):
    VERY_LOW = "very low"
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    VERY_HIGH = "very high"


values_and_translations = [
    ("gender_birth", "Gender", ),
    ("ethnic_group", "Ethnic group"),
    ("working_at_baseline", "Employment status"),
    ("education_age", "Education age"),
    ("smoker", "Smoker"),
    ("iss_category", "Injury Severity Score category"),
    ("penetrating_injury", "Penetrating injury"),
    # int
    ("days_in_hospital", "Number of days in hospital"),
    ("iss_score", "Injury Severity Score"),
    ("nb_of_fractures", "Number of fractures"),
    ("eq5d_5l_q6_i2", "EQ-5D-5L VAS"),
    ("hads_dep_score", "HADS depression"),
    ("bl_cpqs_intensity_average", "Pain intensity (average)"),
    ("bl_cpqs_intensity_now", "Pain intensity (current)"),
    ("bl_cpqs_intensity_worst", "Pain intensity (worst)"),
    ("bl_cpqs_intensity_least", "Pain intensity (least)"),
    ("pain_region_count", "Pain region count"),
    ("bl_sleep_24hrs", "Sleep quality past 24h"),
    ("bl_sleep_quality", "Sleep quality average since injury"),
    # float
    ("age", "Age"),
    ("bmi", "BMI"),
    ("eq5d_5l_index_score", "EQ-5D-5L index"),
    ("sf36_mental_summary", "SF-36 mental summary score"),
    ("sf36_physical_summary", "SF-36 physical summary score"),
    ("hads_anx_score", "HADS anxiety"),
    ("tsk_11_total", "TSK-11 Total"),
    ("pseq_total", "PSEQ Total"),
    ("ies_r_total", "IES-R Total"),
    ("ies_r_avoidance", "IES-R Avoidance"),
    ("ies_r_hyperarousal", "IES-R Hyperarousal"),
    ("ies_r_intrusion", "IES-R Intrusion"),
    ("pe_total_percent_baseline", "Pain extent at baseline"),
    ("paindetect_grand_total", "PainDETECT Total"),
    ("local_heat_average", "Local heat pain threshold"),
    ("remote_heat_average", "Remote heat pain threshold"),
    ("local_cold_average", "Local cold pain threshold"),
    ("remote_cold_average", "Remote cold pain threshold"),
    ("local_pressure_average", "Local pressure pain threshold"),
    ("remote_pressure_average", "Remote pressure pain threshold"),
    ("crp_test", "C-reactive protein (CRP)")
]

# floats and ints
numerical_values = [
    "age",
    "bmi",
    "eq5d_5l_index_score",
    "sf36_mental_summary",
    "sf36_physical_summary",
    "hads_anx_score",
    "tsk_11_total",
    "pseq_total",
    "ies_r_total",
    "ies_r_avoidance",
    "ies_r_hyperarousal",
    "ies_r_intrusion",
    "pe_total_percent_baseline",
    "paindetect_grand_total",
    "local_heat_average",
    "remote_heat_average",
    "local_cold_average",
    "remote_cold_average",
    "local_pressure_average",
    "remote_pressure_average",
    "crp_test",
    # int
    "days_in_hospital",
    "iss_score",
    "nb_of_fractures",
    "eq5d_5l_q6_i2",
    "hads_dep_score",
    "bl_cpqs_intensity_average",
    "bl_cpqs_intensity_now",
    "bl_cpqs_intensity_worst",
    "bl_cpqs_intensity_least",
    "pain_region_count",
    "bl_sleep_24hrs",
    "bl_sleep_quality"
]

categories = [
    "education_age",
    "gender_birth",
    "working_at_baseline",
    "ethnic_group",
    "smoker",
    "iss_category",
    "penetrating_injury"
]


# some values are missing!!!
patient_info_categories = {
    # "General patient characteristics"
    "General characteristics": [
        "age",
        "gender_birth",
        "bmi",
        "ethnic_group",
        "education_age",
        "working_at_baseline",
        "smoker",
        "days_in_hospital"],

    "Injury characteristics": [
        "iss_score",
        "iss_category",
        "nb_of_fractures",
        "penetrating_injury"],

    "Quality of life and physical functioning": [
        "eq5d_5l_index_score",
        "eq5d_5l_q6_i2",
        "sf36_mental_summary",
        "sf36_physical_summary",
        "hads_anx_score",
        "hads_dep_score",
        "tsk_11_total",
        "pseq_total",
        "ies_r_total",
        "ies_r_avoidance",
        "ies_r_hyperarousal",
        "ies_r_intrusion",
        "bl_sleep_24hrs",
        "bl_sleep_quality"],

    "Pain characteristics": [
        "bl_cpgs_intensity_average",
        "bl_cpgs_intensity_now",
        "bl_cpgs_intensity_worst",
        "pain_region_count",
        "pe_total_percent_baseline",
        "bl_cpgs_intensity_least",
        "paindetect_grand_total"],

    "Quantitative sensory testing": [
        "local_heat_average",
        "remote_heat_average",
        "local_cold_average",
        "remote_cold_average",
        "local_pressure_average",
        "remote_pressure_average"],

    "Biomarkers": [
         "crp_test"]
}