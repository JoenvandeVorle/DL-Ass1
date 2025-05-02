class LogLevel:

    LEVEL = 0

    def set_level(level: int) -> None:
        LogLevel.LEVEL = level

    class Level:
        NONE = 0
        ERROR = 1
        WARNING = 2
        INFO = 3
        VERBOSE = 4