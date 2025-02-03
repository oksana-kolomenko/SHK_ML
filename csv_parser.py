from pathlib import Path

import pandas as pd

from categories import GenderBirth, Education, Ethnicity, EmploymentStatus, Smoker, \
    InjurySeverityScoreCategory, PenetratingInjury
from values import values_and_translations, patient_info_categories, categories, numerical_values, \
    Classification


# Get the current working directory
# current_directory = os.getcwd()
# upper_dir = os.path.dirname(os.getcwd())
# file_path = os.path.join(os.path.dirname(os.getcwd()), 'X.csv')
# print(file_path)
file_path = 'X.csv'


tab_data = pd.read_csv(file_path)
values_and_translations_dict = dict(values_and_translations)


def create_patient_summaries():
    df = tab_data
    summaries = []  # "We want to predict health risks. "
    patient_number = 0
    for _, row in df.iterrows():
        patient_number += 1
        patient_info_n_values = row.to_dict()
        summary = (f"The following is the data for patient number {patient_number}." +
                   patient_info(patient_info_n_values))
        summaries.append(summary)
    return summaries

def patient_info(patient_info_n_values):
    patient_info_row = dict(patient_info_n_values)
    patient_info_text = ""

    for patient_info_category, category_values in patient_info_categories.items():
        feature_category_title = f" {patient_info_category} data: "

        category_text = []

        for cat_value in category_values:
            translation = values_and_translations_dict.get(
                cat_value, "Dieser Eigenschaft wurde im Code noch keine Kategorie zugewiesen.")

            value = patient_info_row.get(cat_value, "Not provided")
            if value == "Not provided" or value is None or value == "":
                continue

            translated_value = ""

            if cat_value in categories:
                if isinstance(value, (int, float)) and not pd.isna(value):
                    translated_value = category_to_word(value=value, category_name=cat_value)
                else:
                    translated_value = ""
            elif cat_value in numerical_values:
                if isinstance(value, (int, float)) and not pd.isna(value):
                    average = calculate_average(tab_data, cat_value)
                    std_dev = calculate_standard_deviation(tab_data, cat_value)
                    translated_value = number_to_word(value=value, average=average, std_deviation=std_dev)
                else:
                    translated_value = ""

            if translated_value != "":
                category_text.append(f"{translation} is {translated_value}")

        if category_text:
            patient_info_text += feature_category_title
            patient_info_text += "; ".join(category_text) + "; "
    
    """
        for index, row in tab_data.iterrows():
        structured_text += f"{row['Parameter']} of {row['Value']} "
        structured_text = structured_text.strip(", ") + "."
    """

    return patient_info_text.rstrip('; ') + "."


def calculate_average(df, column_name):
    return df[column_name].mean()


def calculate_standard_deviation(df, column_name):
    return df[column_name].std()


def number_to_word(value, average, std_deviation):
    if value < (average - 2*std_deviation):
        return Classification.VERY_LOW.value
    elif (average - 2*std_deviation) <= value < (average - std_deviation):
        return Classification.LOW.value
    elif (average - std_deviation) <= value < (average + std_deviation):
        return Classification.NORMAL.value
    elif (average + std_deviation) <= value < (average + 2 * std_deviation):
        return Classification.HIGH.value
    elif value >= (average + 2*std_deviation):
        return Classification.VERY_HIGH.value
    # This should never happen
    return "Unclear classification"


def category_to_word(category_name, value):
    """
    Values with category:
    Gender
    Ethnic group
    Education age
    Employment status
    Smoker
    Injury Severity Score category
    Penetrating injury
    Days in hospital???
    iss score?
    nb of fractures?
    usw.
    """

    if isinstance(value, float):
        value = int(value)

    if category_name == "gender_birth":
        for item in GenderBirth:
            if item.value == value:
                return item.name

    elif category_name == "ethnic_group":
        for item in Ethnicity:
            if item.value == value:
                return item.name

    elif category_name == "education_age":
        for item in Education:
            if item.value == value:
                return item.name

    elif category_name == "working_at_baseline":
        for item in EmploymentStatus:
            if item.value == value:
                return item.name

    elif category_name == "smoker":
        for item in Smoker:
            if item.value == value:
                return item.name

    elif category_name == "iss_category":
        for item in InjurySeverityScoreCategory:
            if item.value == value:
                return item.name

    elif category_name == "penetrating_injury":
        for item in PenetratingInjury:
            if item.value == value:
                return item.name
    # This should never happen
    return "Category not found"


with open('Summaries.txt', 'w') as file:
    file.write("\n".join(create_patient_summaries()))
