# OCR Expense Extractor

This repository turns cleaned OCR receipt text into structured expense data.

## Project Structure

- `src/` - Python scripts
- `outputs/` - generated CSV and JSON files from the extractor
- `bills/` - raw receipt images
- `bills_cleaned/` - cleaned receipt images
- `inputs/bills/` - raw receipt images (move your images here)
- `inputs/bills_cleaned/` - cleaned receipt images
- `inputs/bills_cleaned.txt` - OCR text input produced by the OCR processor

## What It Does

- Reads OCR text from `bills_cleaned.txt`
- Sends the text to OpenAI for receipt parsing and OCR word correction
- Normalizes tax data so each item gets a `tax_percentage`, `taxes_and_charges_allocated`, and `final_item_amount`
- Deduplicates repeated bills when OCR splits one receipt into multiple merchant names
- Writes the result to `outputs/expenses_table.csv` and `outputs/expenses_table.json`

## Main Script

Run the extractor with:

```bash
python src/expense_extractor.py
```

Optional arguments:

```bash
python src/expense_extractor.py --input bills_cleaned.txt --csv outputs/expenses_table.csv --json outputs/expenses_table.json --model gpt-4o-mini --max-retries 3
```

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

The extractor also needs an OpenAI API key:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

You can optionally set a model name:

```powershell
$env:OPENAI_MODEL="gpt-4o-mini"
```

## Input And Output

Input:

- `bills_cleaned.txt` - cleaned OCR text to process

Outputs:

- `outputs/expenses_table.csv` - flat item table
- `outputs/expenses_table.json` - structured JSON payload

## Output Fields

Item rows include:

- `item_name`
- `quantity`
- `unit_price`
- `base_amount`
- `tax_percentage`
- `taxes_and_charges_allocated`
- `final_item_amount`

The extractor no longer writes a bill-level final amount field.

## Notes

- OCR text can be noisy, so the prompt asks the model to autocorrect obvious letter errors when the surrounding context makes the intended word clear.
- Duplicate bills are collapsed when the extractor sees the same receipt content under different merchant names.
