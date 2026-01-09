# GenAI Project

This project demonstrates a complete workflow for processing medical notes: extracting structured data using a Large Language Model (LLM) and generating a professional PDF report.

## Workflow Overview

The system operates in two main stages:

### 1. Structured Data Extraction (`main.py`)
The `main.py` script utilizes an LLM (connected via an OpenAI-compatible API) to analyze unstructured medical text. It extracts key information based on a defined Pydantic schema, for example:
- Diagnosis
- Medications
- Allergies
- Follow-up instructions

### 2. PDF Report Generation
Once the structured data is obtained, it is processed to create a final report:
- **Template**: The `files/template` directory contains the HTML template (`template.html`) which defines the report's layout and styling.
- **Rendering**: The processed data is injected into the HTML template.
- **Conversion**: The populated HTML is converted into a PDF file using the `files/render_pdf.py` script.

## Directory Structure

- **`main.py`**: The core script for interfacing with the model to generate structured data.
- **`files/`**:
  - **`template/`**: Stores the HTML template (`template.html`) used for the report.
  - **`data_sample/`**: Contains sample structured data (e.g., `sample.json`).
  - **`render_pdf.py`**: The script responsible for reading the data, populating the template, and rendering the PDF.
  - **`output.pdf`**: The resulting PDF report.

## Usage

To generate a PDF report from the sample data:

```bash
python files/render_pdf.py
```
