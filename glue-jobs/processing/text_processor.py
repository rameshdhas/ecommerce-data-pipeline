from typing import Dict, Any
from shared.utils import safe_float, safe_int


def create_text_content(row: Dict[str, Any]) -> str:
    """
    Create text content for embedding generation optimized for e-commerce semantic search
    """
    text_parts = []

    # Primary fields - most important for search
    if 'title' in row and row['title']:
        text_parts.append(f"Product: {row['title']}")

    if 'description' in row and row['description']:
        text_parts.append(f"Description: {row['description']}")

    if 'brand' in row and row['brand']:
        text_parts.append(f"Brand: {row['brand']}")

    if 'categories' in row and row['categories']:
        # Categories might be a list or string
        categories = row['categories']
        if isinstance(categories, str):
            text_parts.append(f"Categories: {categories}")
        else:
            text_parts.append(f"Categories: {', '.join(str(c) for c in categories)}")

    if 'features' in row and row['features']:
        features = row['features']
        if isinstance(features, str):
            text_parts.append(f"Features: {features}")
        else:
            text_parts.append(f"Features: {', '.join(str(f) for f in features)}")

    # Secondary fields - additional context
    if 'department' in row and row['department']:
        text_parts.append(f"Department: {row['department']}")

    if 'manufacturer' in row and row['manufacturer']:
        text_parts.append(f"Manufacturer: {row['manufacturer']}")

    if 'product_details' in row and row['product_details']:
        text_parts.append(f"Details: {row['product_details']}")

    if 'variations' in row and row['variations']:
        text_parts.append(f"Variations: {row['variations']}")

    # Context fields - enrich with signals
    if 'rating' in row and row['rating']:
        rating = safe_float(row['rating'])
        if rating >= 4.5:
            text_parts.append("Highly rated product")
        elif rating >= 4.0:
            text_parts.append("Well rated product")

    if 'reviews_count' in row and row['reviews_count']:
        reviews = safe_int(row['reviews_count'])
        if reviews > 1000:
            text_parts.append("Popular product with many reviews")
        elif reviews > 100:
            text_parts.append("Well-reviewed product")

    if 'availability' in row and row['availability']:
        if 'in stock' in str(row['availability']).lower():
            text_parts.append("Currently available")

    if 'discount' in row and row['discount']:
        discount_str = str(row['discount']).replace('%', '').strip('"\'')
        discount = safe_float(discount_str)
        if discount >= 50:
            text_parts.append("Significant discount available")
        elif discount >= 20:
            text_parts.append("On sale")

    # Price range information
    if 'final_price' in row and row['final_price']:
        text_parts.append(f"Price: {row['final_price']} {row.get('currency', 'USD')}")

    return " | ".join(text_parts)


def extract_clean_description(row: Dict[str, Any]) -> str:
    """
    Extract a clean product description from the raw data.
    This handles cases where the description field contains JSON or concatenated data.
    """
    description = row.get('description', '')

    if not description:
        return ''

    # If description contains JSON-like structure, try to extract just the description part
    if 'Description:' in description:
        # Extract the description part from concatenated text
        try:
            # Find the Description: part and extract until the next field
            desc_start = description.find('Description: ') + len('Description: ')
            desc_end = description.find(' | ', desc_start)
            if desc_end == -1:
                desc_end = len(description)
            return description[desc_start:desc_end].strip()
        except Exception:
            pass

    # If it looks like JSON, try to parse it
    if description.strip().startswith('{') or 'Product:' in description:
        # This appears to be the concatenated product data, return empty for now
        # The actual description should be extracted from the individual fields
        return ''

    # Otherwise return the description as-is
    return description.strip()