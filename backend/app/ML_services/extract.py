import re

def extract_age(text: str):
    match = re.search(r"\b(\d{2})\s*years?\s*old\b|\bage\s*[:\-]?\s*(\d{2})", text, re.I)
    if match:
        return int(match.group(1) or match.group(2))
    return None

def extract_salary(text: str):
    match = re.search(r"salary\s*[:\-]?\s*(\d{4,7})|\b(\d{4,7})\s*(?:per\s*month|monthly)", text, re.I)
    if match:
        return int(match.group(1) or match.group(2))
    return None

def extract_loan_amount(text: str):
    match = re.search(r"loan\s*amount\s*[:\-]?\s*(\d{4,7})|\bneed\s*(\d{4,7})\b", text, re.I)
    if match:
        return int(match.group(1) or match.group(2))
    return None

def extract_employment_type(text: str):
    text = text.lower()
    if "self" in text or "own business" in text:
        return "self-employed"
    if "business" in text:
        return "business"
    if "student" in text:
        return "student"
    if "unemployed" in text:
        return "unemployed"
    if "job" in text or "company" in text or "working" in text:
        return "salaried"
    return None

def extract_features(text: str):
    return {
        "age": extract_age(text),
        "salary": extract_salary(text),
        "loan_amount": extract_loan_amount(text),
        "employment_type": extract_employment_type(text),
    }
