class TaxationDataWarning(Warning):
    """Raised when a taxation tree is present under the raw-data root but its
    personal-income-tax schedules are missing.

    Mirrors the ``DataFilterWarning`` convention used elsewhere in the data
    pipeline.  Unlike the optional energy-sector readers (which skip silently),
    a ``taxation/`` directory that exists but lacks its schedules is treated as a
    misconfiguration and surfaced, while the wholesale absence of a ``taxation/``
    directory stays silent (taxation simply not in use).
    """

    pass


# Re-exported for convenience.  Imported lazily by consumers that must stay
# import-light (e.g. the configurations-layer builder references it only under
# TYPE_CHECKING); ``DataReaders.from_raw_data`` imports it directly.
from macro_data.readers.taxation.taxation_reader import TaxationReader  # noqa: E402

__all__ = ["TaxationDataWarning", "TaxationReader"]
