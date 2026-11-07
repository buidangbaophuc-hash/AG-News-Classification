"""AG News classification package."""
import sys

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure') and getattr(_stream, 'encoding', None):
        if _stream.encoding.lower().replace('-', '') != 'utf8':
            try:
                _stream.reconfigure(encoding='utf-8', errors='replace')
            except (AttributeError, ValueError, OSError):
                pass
