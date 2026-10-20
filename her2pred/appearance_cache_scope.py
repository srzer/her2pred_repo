"""Check appearance targets for the records that actually use that loss."""

from parse_slide_ids import norm_key


def validate_coverage(train_records, val_records, dom_of, selected_ids,
                      missing_ids, external_site_only=False,
                      site_only_adv=False):
    if site_only_adv:
        return

    def uses_appearance(record):
        return (not external_site_only
                or str(record.get('source', '')) == 'ndpi')

    active_train = [record for record in train_records
                    if uses_appearance(record)]
    missing = [record for record in active_train
               if dom_of.get(record['slide_id'], -1) < 0]

    if selected_ids is not None:
        cohort = {
            norm_key(str(record['slide_id']))
            for record in list(train_records) + list(val_records)
            if not record.get('adapt_only') and uses_appearance(record)
        }
        if cohort != selected_ids:
            raise RuntimeError(
                'appearance-cache cohort mismatch: PT slide_id values for '
                'appearance-supervised sources do not equal the cache '
                'selected_sid declaration'
            )
        actual_missing = {norm_key(str(record['slide_id']))
                          for record in missing}
        train_ids = {norm_key(str(record['slide_id']))
                     for record in active_train}
        if actual_missing != (missing_ids & train_ids):
            raise RuntimeError(
                'appearance-cache missing-target pattern differs from the '
                'frozen declared pattern'
            )
    elif missing:
        by_source = {}
        for record in missing:
            source = str(record.get('source', 'unknown'))
            by_source[source] = by_source.get(source, 0) + 1
        raise RuntimeError(
            'appearance-cache coverage failure: '
            f'{len(missing)}/{len(active_train)} appearance-supervised '
            f'optimization records are missing targets '
            f'({dict(sorted(by_source.items()))}). '
            'The cache sid values must match the slide_id values stored '
            'inside the PT tensors.'
        )
