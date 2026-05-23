import argparse
import csv
import json
import os
import time
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

import requests


INPUT_FILE = "bills_cleaned.txt"
OUTPUT_CSV = "expenses_table.csv"
OUTPUT_JSON = "expenses_table.json"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_MAX_RETRIES = 3


EXPENSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["bills"],
    "properties": {
        "bills": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "source_file",
                    "restaurant_name",
                    "date",
                    "currency",
                    "items",
                    "taxes_and_charges",
                    "subtotal",
                    "final_bill_amount",
                    "confidence",
                    "warnings",
                ],
                "properties": {
                    "source_file": {"type": ["string", "null"]},
                    "restaurant_name": {"type": ["string", "null"]},
                    "date": {"type": ["string", "null"]},
                    "currency": {"type": ["string", "null"]},
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "name",
                                "quantity",
                                "unit_price",
                                "base_amount",
                                "taxes_and_charges_allocated",
                                "final_item_cost",
                                "notes",
                            ],
                            "properties": {
                                "name": {"type": "string"},
                                "quantity": {"type": ["number", "null"]},
                                "unit_price": {"type": ["number", "null"]},
                                "base_amount": {"type": ["number", "null"]},
                                "taxes_and_charges_allocated": {"type": ["number", "null"]},
                                "final_item_cost": {"type": ["number", "null"]},
                                "notes": {"type": ["string", "null"]},
                            },
                        },
                    },
                    "taxes_and_charges": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["name", "amount"],
                            "properties": {
                                "name": {"type": "string"},
                                "amount": {"type": ["number", "null"]},
                            },
                        },
                    },
                    "subtotal": {"type": ["number", "null"]},
                    "final_bill_amount": {"type": ["number", "null"]},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "warnings": {"type": "array", "items": {"type": "string"}},
                },
            },
        }
    },
}


def money(value):
    if value is None or value == "":
        return ""
    try:
        return str(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    except (InvalidOperation, ValueError):
        return str(value)


def extract_output_text(response_json):
    if response_json.get("output_text"):
        return response_json["output_text"]

    chunks = []
    for output in response_json.get("output", []):
        for content in output.get("content", []):
            if content.get("type") == "output_text" and "text" in content:
                chunks.append(content["text"])
    return "".join(chunks)


def get_api_error_message(response):
    try:
        error = response.json().get("error", {})
    except ValueError:
        return response.text.strip()

    message = error.get("message") or response.text.strip()
    error_type = error.get("type")
    error_code = error.get("code")

    details = []
    if error_type:
        details.append(f"type={error_type}")
    if error_code:
        details.append(f"code={error_code}")

    if details:
        return f"{message} ({', '.join(details)})"
    return message


def raise_for_api_error(response):
    if response.status_code != 429:
        raise RuntimeError(
            f"OpenAI API error {response.status_code}: {get_api_error_message(response)}"
        )

    message = get_api_error_message(response)
    lower_message = message.lower()
    if "insufficient_quota" in lower_message or "quota" in lower_message:
        raise RuntimeError(
            "OpenAI API quota/billing limit reached. Check your account billing, "
            f"project limits, or try a different API key. API message: {message}"
        )

    raise RuntimeError(
        "OpenAI API rate limit reached after retries. Wait a minute and run again, "
        "or use a smaller input/model. "
        f"API message: {message}"
    )


def retry_delay(response, attempt):
    retry_after = response.headers.get("retry-after")
    if retry_after:
        try:
            return max(1.0, float(retry_after))
        except ValueError:
            pass
    return min(30.0, 2.0**attempt)


def call_openai(extracted_text, model, max_retries):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. In PowerShell, run: "
            "$env:OPENAI_API_KEY='your_api_key_here'"
        )

    instructions = (
        "You extract restaurant expense data from noisy OCR receipt text. "
        "Return only data supported by the OCR. Fix obvious OCR mistakes when context is clear. "
        "For each item, keep base_amount as the receipt line amount before bill-level taxes. "
        "Set taxes_and_charges_allocated to the item's proportional share of taxes, service charges, "
        "rounding adjustments, or other bill-level additions. Set final_item_cost to base_amount plus "
        "that allocation, so item final costs add up as closely as possible to final_bill_amount. "
        "If the OCR is unclear, use nulls and add a warning."
    )

    payload = {
        "model": model,
        "instructions": instructions,
        "input": extracted_text,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "restaurant_expenses",
                "strict": True,
                "schema": EXPENSE_SCHEMA,
            }
        },
    }

    for attempt in range(max_retries + 1):
        response = requests.post(
            OPENAI_RESPONSES_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )

        if response.ok:
            break

        if response.status_code != 429 or attempt == max_retries:
            raise_for_api_error(response)

        delay = retry_delay(response, attempt)
        print(
            f"Rate limited by OpenAI. Retrying in {delay:.0f} seconds "
            f"({attempt + 1}/{max_retries})..."
        )
        time.sleep(delay)

    output_text = extract_output_text(response.json())
    if not output_text:
        raise RuntimeError("The API response did not contain output text.")
    return json.loads(output_text)


def write_json(data, output_json):
    with open(output_json, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def write_csv(data, output_csv):
    columns = [
        "source_file",
        "restaurant_name",
        "date",
        "currency",
        "item_name",
        "quantity",
        "unit_price",
        "base_amount",
        "taxes_and_charges_allocated",
        "final_item_cost",
        "final_bill_amount",
        "confidence",
        "warnings",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for bill in data.get("bills", []):
            warnings = " | ".join(bill.get("warnings", []))
            for item in bill.get("items", []):
                writer.writerow(
                    {
                        "source_file": bill.get("source_file") or "",
                        "restaurant_name": bill.get("restaurant_name") or "",
                        "date": bill.get("date") or "",
                        "currency": bill.get("currency") or "",
                        "item_name": item.get("name") or "",
                        "quantity": item.get("quantity") or "",
                        "unit_price": money(item.get("unit_price")),
                        "base_amount": money(item.get("base_amount")),
                        "taxes_and_charges_allocated": money(
                            item.get("taxes_and_charges_allocated")
                        ),
                        "final_item_cost": money(item.get("final_item_cost")),
                        "final_bill_amount": money(bill.get("final_bill_amount")),
                        "confidence": bill.get("confidence") or "",
                        "warnings": warnings,
                    }
                )


def main():
    parser = argparse.ArgumentParser(
        description="Use the OpenAI API to convert OCR receipt text into an expense table."
    )
    parser.add_argument("--input", default=INPUT_FILE, help="OCR text file to read.")
    parser.add_argument("--csv", default=OUTPUT_CSV, help="CSV table to write.")
    parser.add_argument("--json", default=OUTPUT_JSON, help="Structured JSON to write.")
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        help="OpenAI model to use. Defaults to OPENAI_MODEL or gpt-5.4-mini.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Number of retries for temporary OpenAI rate limits.",
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as file:
        extracted_text = file.read().strip()

    if not extracted_text:
        raise RuntimeError(f"No OCR text found in {args.input}.")

    data = call_openai(extracted_text, args.model, args.max_retries)
    write_json(data, args.json)
    write_csv(data, args.csv)

    print(f"Wrote {args.csv}")
    print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
