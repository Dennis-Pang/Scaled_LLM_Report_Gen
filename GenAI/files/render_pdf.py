import json
import os
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML

# Define paths
BASE_DIR = '/home/zixian.pang/GenAI/files'
DATA_FILE = os.path.join(BASE_DIR, 'data_sample/sample.json')
TEMPLATE_DIR = os.path.join(BASE_DIR, 'template')
TEMPLATE_FILE = 'template.html'
OUTPUT_FILE = os.path.join(BASE_DIR, 'output.pdf')

def main():
    # 1. Read data
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded data from {DATA_FILE}")
    except Exception as e:
        print(f"Error reading data file: {e}")
        return

    # 2. Setup Jinja2 environment
    try:
        env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
        template = env.get_template(TEMPLATE_FILE)
        print(f"Loaded template from {os.path.join(TEMPLATE_DIR, TEMPLATE_FILE)}")
    except Exception as e:
        print(f"Error loading template: {e}")
        return

    # 3. Render HTML
    try:
        rendered_html = template.render(**data)
        print("Template rendered successfully")
    except Exception as e:
        print(f"Error rendering template: {e}")
        return
    
    # 4. Generate PDF
    try:
        # base_url is set to TEMPLATE_DIR to resolve any relative paths in the HTML (css, images, etc.)
        HTML(string=rendered_html, base_url=TEMPLATE_DIR).write_pdf(OUTPUT_FILE)
        print(f"PDF generated successfully: {OUTPUT_FILE}")
    except Exception as e:
        print(f"Error generating PDF: {e}")

if __name__ == "__main__":
    main()
