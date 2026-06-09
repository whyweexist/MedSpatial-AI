"""Provenance graph utilities."""

from app.awm.schema import ProvenanceRecord


def lineage(records: list[ProvenanceRecord], output_id: str) -> list[ProvenanceRecord]:
    by_output = {output: record for record in records for output in record.output_ids}
    result: list[ProvenanceRecord] = []
    pending = [output_id]
    seen: set[str] = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        record = by_output.get(current)
        if record:
            result.append(record)
            pending.extend(record.source_ids)
    return result
