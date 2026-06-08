"""Label maps: model-native NE labels -> the 10-label spec scheme (#3/#23).

Used to score OPF and GiNZA against the same gold with one criterion.
None means the native label has no spec counterpart (dropped in typed scoring).
Mapping is approximate and documented as such in the report.
"""
from __future__ import annotations

SPEC_LABELS = ["PERSON", "ADDRESS", "PHONE", "EMAIL", "DATE", "ID",
               "AGE", "REGION", "OCCUPATION", "ORGANIZATION"]

# OpenAI Privacy Filter default v2 categories (8)
OPF_TO_SPEC = {
    "private_person": "PERSON",
    "private_address": "ADDRESS",
    "private_phone": "PHONE",
    "private_email": "EMAIL",
    "private_date": "DATE",
    "account_number": "ID",
    "private_url": None,   # no spec counterpart
    "secret": None,
}

# GiNZA (ja_ginza) extended-NE labels -> spec (approximate).
# Administrative areas -> REGION (quasi-identifier); GiNZA fragments full
# addresses so ADDRESS rarely matches (an honest, expected weakness).
GINZA_TO_SPEC = {
    "Person": "PERSON", "Name_Other": "PERSON",
    "Province": "REGION", "City": "REGION", "County": "REGION",
    "GPE": "REGION", "Geological_Region_Other": "REGION", "Region_Other": "REGION",
    "Address": "ADDRESS",
    "Phone_Number": "PHONE", "Email": "EMAIL",
    "Date": "DATE", "Time": "DATE", "Era": "DATE", "Day_Of_Week": "DATE",
    "Age": "AGE",
    "Position_Vocation": "OCCUPATION", "Occupation_Title": "OCCUPATION",
    "Company": "ORGANIZATION", "Corporation_Other": "ORGANIZATION",
    "Organization_Other": "ORGANIZATION", "Company_Group": "ORGANIZATION",
    "School": "ORGANIZATION", "Institution": "ORGANIZATION",
    "Medical_Institution": "ORGANIZATION", "Government": "ORGANIZATION",
    "Show_Organization": "ORGANIZATION", "International_Organization": "ORGANIZATION",
    "Public_Institution": "ORGANIZATION",
    "ID_Number": "ID",
}
