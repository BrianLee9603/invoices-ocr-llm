import numpy as np
from typing import List, Optional
from src.schemas.document import TextBlock
from src.services.processing.ocr.postprocessing.post_processor import OcrPostProcessor


class LayoutReconstructor:
    """Reconstruct document layout from OCR text blocks using spatial clustering.

    Takes scattered OCR blocks with bbox coordinates and reconstructs the
    original reading order, grouping text into rows and detecting table regions.
    This is critical for invoice extraction because PaddleOCR returns blocks
    in arbitrary order, destroying table structure.

    The output is a formatted string with section markers (HEADER / TABLE / SUMMARY)
    and markdown-style table formatting for tabular regions.
    """

    def __init__(self, row_tolerance_ratio: float = 0.5):
        """
        Args:
            row_tolerance_ratio: How much vertical overlap (as a fraction of
                average block height) is needed to consider two blocks on the
                same row. Higher = more lenient grouping.
        """
        self._row_tolerance_ratio = row_tolerance_ratio

    def reconstruct(
        self,
        text_blocks: List[TextBlock],
        post_processor: Optional[OcrPostProcessor] = None,
        layout_regions: Optional[List[dict]] = None
    ) -> str:
        """Cluster blocks into rows, detect tables, return layout-formatted text.

        Args:
            text_blocks: OCR text blocks with bbox coordinates.
            post_processor: Optional post-processor to fix OCR errors.
            layout_regions: Optional layout regions detected by DocLayout-YOLO.

        Returns:
            Layout-reconstructed text with section markers and table formatting.
        """
        if not text_blocks:
            return ""

        # Filter blocks with valid bboxes
        valid_blocks = []
        for block in text_blocks:
            text = block.text.strip()
            if not text:
                continue
            if post_processor:
                text = post_processor.fix_text(text, block.confidence)
            if block.bbox and len(block.bbox) >= 4:
                valid_blocks.append({
                    'text': text,
                    'confidence': block.confidence,
                    'x_min': block.bbox[0],
                    'y_min': block.bbox[1],
                    'x_max': block.bbox[2],
                    'y_max': block.bbox[3],
                })
            else:
                valid_blocks.append({
                    'text': text,
                    'confidence': block.confidence,
                    'x_min': 0,
                    'y_min': 0,
                    'x_max': 0,
                    'y_max': 0,
                })

        if not valid_blocks:
            return ""

        # Step 1: Cluster blocks into rows by Y coordinate
        rows = self._cluster_into_rows(valid_blocks)

        # Step 2: Detect table vs non-table regions
        sections = self._detect_sections(rows, layout_regions)

        # Step 3: Format output
        return self._format_sections(sections)

    def _cluster_into_rows(self, blocks: list) -> list:
        """Group text blocks into rows based on vertical position overlap."""
        # Sort by y_min (top of block)
        sorted_blocks = sorted(blocks, key=lambda b: b['y_min'])

        # Calculate average block height for tolerance
        heights = [b['y_max'] - b['y_min'] for b in sorted_blocks if b['y_max'] > b['y_min']]
        avg_height = sum(heights) / len(heights) if heights else 20
        tolerance = avg_height * self._row_tolerance_ratio

        rows = []
        current_row = [sorted_blocks[0]]
        current_y_center = (sorted_blocks[0]['y_min'] + sorted_blocks[0]['y_max']) / 2

        for block in sorted_blocks[1:]:
            block_y_center = (block['y_min'] + block['y_max']) / 2

            if abs(block_y_center - current_y_center) <= tolerance:
                # Same row
                current_row.append(block)
                # Use median to be robust against outliers (wavy text lines)
                y_centers = [(b['y_min'] + b['y_max']) / 2 for b in current_row]
                current_y_center = float(np.median(y_centers))
            else:
                # New row — save current and start new
                current_row.sort(key=lambda b: b['x_min'])  # Left-to-right within row
                rows.append(current_row)
                current_row = [block]
                current_y_center = block_y_center

        # Don't forget the last row
        current_row.sort(key=lambda b: b['x_min'])
        rows.append(current_row)

        return rows

    def _detect_sections(self, rows: list, layout_regions: Optional[list] = None) -> list:
        """Detect HEADER, TABLE, and SUMMARY sections based on row structure and YOLO layout analysis."""
        sections = []
        
        yolo_tables = []
        if layout_regions:
            yolo_tables = [r for r in layout_regions if r.get('label') == 'table']
        
        # Analyze column count distribution to find table regions
        table_start = None
        table_header_row = None
        in_table = False
        consecutive_multi_col = 0

        for i, row in enumerate(rows):
            num_cols = len(row)
            
            # Calculate row y center to check overlap with YOLO table regions
            y_centers = [(b['y_min'] + b['y_max']) / 2 for b in row]
            row_y_center = float(np.median(y_centers)) if y_centers else 0
            
            is_yolo_table_row = False
            for tbl in yolo_tables:
                tbl_box = tbl['bbox'] # [xmin, ymin, xmax, ymax]
                # Allow a small vertical tolerance (5% of table height)
                tolerance = (tbl_box[3] - tbl_box[1]) * 0.05
                if (tbl_box[1] - tolerance) <= row_y_center <= (tbl_box[3] + tolerance):
                    is_yolo_table_row = True
                    break
            
            # Check for section markers
            row_text = ' '.join(b['text'] for b in row).strip().upper()
            
            if row_text in ('ITEMS', 'SUMMARY', 'TOTAL'):
                # Explicit section boundary
                if in_table:
                    sections.append({
                        'type': 'table',
                        'header_row': table_header_row,
                        'rows': rows[table_start:i],
                    })
                    in_table = False
                    consecutive_multi_col = 0
                sections.append({'type': 'marker', 'text': row_text, 'row_index': i})
                continue

            # Tabular check: either YOLO table row OR heuristic tabular row
            is_tabular = is_yolo_table_row or (num_cols >= 2 and self._is_tabular_row(row))

            if is_tabular:
                consecutive_multi_col += 1
                if consecutive_multi_col >= 2 and not in_table:
                    # Start of a table region (retroactively include first multi-col row)
                    in_table = True
                    table_start = i - consecutive_multi_col + 1
                    table_header_row = rows[table_start] if table_start < len(rows) else None
            else:
                if in_table and consecutive_multi_col < 2:
                    # Single-col row in table context — might be a multi-line description
                    # Keep it in the table if surrounded by table rows
                    pass
                elif in_table:
                    sections.append({
                        'type': 'table',
                        'header_row': table_header_row,
                        'rows': rows[table_start:i],
                    })
                    in_table = False
                    consecutive_multi_col = 0
                else:
                    consecutive_multi_col = 0

            if not in_table:
                sections.append({'type': 'text', 'row': row, 'row_index': i})

        # Close any open table
        if in_table:
            sections.append({
                'type': 'table',
                'header_row': table_header_row,
                'rows': rows[table_start:],
            })

        return sections

    def _is_tabular_row(self, row: list) -> bool:
        """
        Check if a row has tabular structure (not just 2 text blocks on same line).
        
        For rows with >= 3 columns, always return True.
        For rows with exactly 2 columns, check if there's meaningful horizontal
        separation (gap > 15% of row width between blocks).
        """
        if len(row) >= 3:
            return True
        
        if len(row) == 2:
            # Check gap between the two blocks
            block_a, block_b = row[0], row[1]
            gap = block_b['x_min'] - block_a['x_max']
            row_width = block_b['x_max'] - block_a['x_min']
            
            if row_width > 0:
                gap_ratio = gap / row_width
                # Large gap suggests tabular layout, not flowing text
                return gap_ratio > 0.15
        
        return False

    def _format_sections(self, sections: list) -> str:
        """Format detected sections into readable structured text."""
        output_lines = []
        current_section = "HEADER"

        for section in sections:
            if section['type'] == 'marker':
                marker_text = section['text']
                if 'ITEM' in marker_text:
                    current_section = "ITEMS"
                elif 'SUMMAR' in marker_text:
                    current_section = "SUMMARY"
                output_lines.append(f"\n--- {marker_text} ---")

            elif section['type'] == 'table':
                # Format as markdown-style table
                table_rows = section['rows']
                if not table_rows:
                    continue

                # Determine max columns
                max_cols = max(len(r) for r in table_rows)

                # Build table
                formatted_rows = []
                for row in table_rows:
                    cells = [b['text'] for b in row]
                    # Pad to max_cols
                    while len(cells) < max_cols:
                        cells.append('')
                    formatted_rows.append(cells)

                # Calculate column widths
                col_widths = [0] * max_cols
                for row_cells in formatted_rows:
                    for j, cell in enumerate(row_cells):
                        col_widths[j] = max(col_widths[j], len(cell))

                # Output with alignment
                for k, row_cells in enumerate(formatted_rows):
                    line = '| ' + ' | '.join(
                        cell.ljust(col_widths[j]) for j, cell in enumerate(row_cells)
                    ) + ' |'
                    output_lines.append(line)
                    # Add separator after first row (header)
                    if k == 0:
                        sep = '|' + '|'.join(
                            '-' * (col_widths[j] + 2) for j in range(max_cols)
                        ) + '|'
                        output_lines.append(sep)

            elif section['type'] == 'text':
                row = section['row']
                # Join blocks in the row with appropriate spacing
                line = '    '.join(b['text'] for b in row)
                output_lines.append(line)

        return '\n'.join(output_lines)
