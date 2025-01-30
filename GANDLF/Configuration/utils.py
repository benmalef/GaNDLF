from typing import Type
from pydantic import BaseModel

from typing import Type
from pydantic import BaseModel

def generate_and_save_markdown(model: Type[BaseModel], file_path: str) -> None:
    schema = model.schema()
    markdown = []

    # Add title
    markdown.append(f"# {schema['title']}\n")

    # Add description if available
    if "description" in schema:
        markdown.append(f"{schema['description']}\n")

    # Add fields table
    markdown.append("## Parameters\n")
    markdown.append("| Field | Type | Description | Default |")
    markdown.append("|----------------|----------------|-----------------------|------------------|")

    for field_name, field_info in schema["properties"].items():
        # Extract field details
        field_type = field_info.get("type", "N/A")
        description = field_info.get("description", "N/A")
        default = field_info.get("default", "N/A")

        # Add row to the table
        markdown.append(f"| `{field_name}` | `{field_type}` | {description} | `{default}` |")

    # Write to file
    with open(file_path, "w", encoding="utf-8") as file:
        file.write("\n".join(markdown))