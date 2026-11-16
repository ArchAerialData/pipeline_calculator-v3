"""Conservative import budgets, independent of the analysis workload budget."""
MAX_SOURCE_BYTES = 256 * 1024 * 1024
MAX_EXPANDED_BYTES = 256 * 1024 * 1024
MAX_ENTRIES = 10_000
MAX_FEATURES = 250_000
MAX_VERTICES = 2_000_000
MAX_COLLECTION_DEPTH = 32
MAX_ATTRIBUTE_BYTES = 32 * 1024 * 1024
MAX_DIAGNOSTICS = 10_000
MAX_LAYERS = 512
CHUNK_BYTES = 1024 * 1024
BATCH_FEATURES = 256
MAX_IPC_BYTES = 64 * 1024 * 1024


def check(context):
    if context is not None:
        context.check()


def report(context, stage, completed=0, total=None):
    if context is not None:
        context.report(stage, completed, total)
