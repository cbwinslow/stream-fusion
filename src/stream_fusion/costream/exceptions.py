"""Custom exceptions for Co-Stream & Cross-Platform Alignment Subsystem (Spec 23)."""


class CoStreamError(Exception):
    """Base exception for all co-stream errors."""
    pass


class StreamSynchronizationError(CoStreamError):
    """Raised when cross-stream alignment or clock synchronization fails."""
    pass


class StreamBufferOverflowError(CoStreamError):
    """Raised when a stream's bounded buffer exceeds safety thresholds."""
    pass


class ChannelConnectionError(CoStreamError):
    """Raised when a co-stream channel fails to connect or reconnects too many times."""
    pass


class ResourceCeilingExceededError(CoStreamError):
    """Raised when session resource usage exceeds configured memory/buffer limits."""
    pass
